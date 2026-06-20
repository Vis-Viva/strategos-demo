# distutils: language = c
# cython: language_level 3
# cython: profile = False


cimport cython
from cython      cimport typeof
from cython.view cimport array as cyarr

cimport numpy as cnp
cnp.import_array()

from strategos_tools.env.player_ops cimport Are_Opponents
from strategos_tools.utils.funcs    cimport *
from strategos_tools.AIOps.models   import  AdvNet, MultiModel
from strategos_tools.AIOps.nn_utils import  AdvNetCompiler, MMCompiler

import pickle, numpy as np
import torch as pt

from time       import time as TimeNow
from numpy      import asarray as NP, ascontiguousarray as CONTIG, expand_dims as newaxis, swapaxes
from numpy      import float64 as f64, float32 as f32, uintc, intc
from torch      import inference_mode as nn_eval_mode, float32 as tf32
from torch.cuda import empty_cache as empty_GPU_cache


# ==================================================================================================
# This module implements all the operations and math necessary for calling our estimator models to
# get adv estimates and perform transformations on them to turn them into action probabilities. 
# Global singleton model instances also live here. 
# ==================================================================================================


# ==================================================================================================
# GLOBAL MODEL INSTANTIATORS
# ==================================================================================================
# These are singletons so they can be directly called from within this module or externally.


cdef void setup_advnet( str modelFile, uint modelIter, uint modelSize=64, uint GPUrank=0, bint Compiled=FALSE ): #noexcept:
	cdef object aNet = AdvNet( modelSize=modelSize, modelIter=modelIter, load_from_file=modelFile )
	aNet.to( f"cuda:{GPUrank}" )
	global ADVNET
	ADVNET = aNet #if not compiled else AdvNetCompiler( aNet,GPUrank )

cdef void setup_multimodel( str modelFile, uint iterSpan, uint modelSize=64, uint GPUrank=0, bint Compiled=FALSE ): #noexcept:
	cdef object MM = MultiModel( modelSize=modelSize, mFile=modelFile, iterSpan=iterSpan, GPUrank=GPUrank )
	global MULTIMODEL
	MULTIMODEL = MM #if not compiled else MMCompiler( MM,iterSpan,GPUrank )

# Having two multimodel singletons allows us to evaluate one test model against another
cdef void setup_alt_multimodel( str modelFile, uint iterSpan, uint modelSize=64, uint GPUrank=0, bint Compiled=FALSE ): #noexcept:
	cdef object aMM = MultiModel( modelSize=modelSize, mFile=modelFile, iterSpan=iterSpan, GPUrank=GPUrank )
	global ALT_MULTIMODEL
	ALT_MULTIMODEL = aMM #if not compiled else MMCompiler( aMM,iterSpan,GPUrank )


# ==================================================================================================
# SINGLE-ITER ADVNET OPS
# ==================================================================================================


# Simple: get formatted inputs ⟶ run them through ADVNET ⟶ return
cdef flt1 __AdvEstimator( infoset I, int GPUrank=0 ): #noexcept:

	cdef AdvNetInputs inputs = AdvNetInputs( I, GPUrank )
	with nn_eval_mode():
		return ADVNET( inputs.H,
					   inputs.hC_c, inputs.hC_r, inputs.hC_s,
					   inputs.fC_c, inputs.fC_r, inputs.fC_s,
					   inputs.tC_c, inputs.tC_r, inputs.tC_s,
					   inputs.rC_c, inputs.rC_r, inputs.rC_s,
					   inputs.A,    inputs.nA ).detach().to( device='cpu',dtype=tf32 ).numpy().ravel()
	#                                           ￪￪￪       This just does GPUtensor ⟶ CPUflt1       ￪￪￪

# [ ∃ a:α(a)>0 ] ⇒ [( σ(a)∝α(a) ∀ a:α(a)>0 ) & ( σ(a)=0 ∀ a:α(a)≤0 )]. [ α(a)≤0 ∀ a ] ⇒ [ σ=𝓤({ a|α(a)=max(α(A)) }) ]
cdef flt1 __ActionProbs( flt1 advI ): #noexcept: # advI shape = (|A(I)|,)

	# TODO: Do we want to keep this short-circuit?
	#if advI.shape[0]==1:
		#return np.ones_like( advI ) # |A|=1 ⇒ σ( a ) = 1 for the only a ∈ A

	# ( ∃ a∈A(I):α(a)>0 ) ⇒ ( σ(a)=α(a)/Σ(α+) ∀a:α(a)>0; σ(a)=0 o.w. ), α+ = { max( α(a),0 ) | a∈A(I) }
	cdef flt1 aPos = ArrClip1d( advI, minVal=0 )
	if (NP( advI )>0).any(): return ArrScale1d( aPos, 1/ArrSum1d( aPos ) )

	# ( α(a)≤0 ∀ a∈A(I) ) ⇒ ( σ(a)=1/|Amax| ∀a:α(a)=max(α(A)); σ(a)=0 o.w. ), Amax = { a|α(a)=max(α(A)) }
	cdef float amax = ArrMax1d( advI )
	cdef uint1 Amax = NP( NP( advI )==amax, dtype=uintc )
	return NP( Amax,dtype=f32 ) / np.count_nonzero( Amax )

# Estimate α(I) ⟶ derive σ(I) as above; used during tree traversal to strategy-sample opponent actions
cdef flt1   Strat( infoset I, int GPUrank=0 ): #noexcept:
	cdef flt1 advI = __AdvEstimator( I, GPUrank )
	return __ActionProbs( advI )


# ==================================================================================================
# MANY-ITER MULTIMODEL OPS
# ==================================================================================================
# Since A(Iᵢ)=A(Iⱼ) ∀Iᵢ,Iⱼ∈𝓘 wherever any multiops are used, henceforth A:=A(I₀)


# Transforms legacy MM outputs from list to flt3. Legacy use only; unusued in modern infrastructure
cdef flt3 __MultiAdvArray( list rawOutputs, uint T, uint nI, uint nA ): #noexcept:
	return CONTIG( NP( rawOutputs,dtype=f32 ).reshape(T,nI,nA) )

# Another legacy helper: turns GPU tensor model outputs into a list on the CPU
cdef list __tolist( object MMoutputs ): #noexcept:
	if type( MMoutputs )==list:
		return MMoutputs
	else:
		return MMoutputs.detach().to( device='cpu',dtype=tf32 ).numpy().ravel().tolist()

# Enables us to do evaluations against legacy models with old input/output formatting
cdef flt3 __LegacyMultiAdvEstimator( uint actingPlayer, infoset I, uint GPUrank=0, bint Alt_Model=FALSE ): #noexcept:
	
	cdef MMInputs_old inputs = MMInputs_old( actingPlayer, I, MULTIMODEL.IterSpan, GPUrank )
	cdef object       outputs

	with nn_eval_mode():
		outputs = MULTIMODEL( inputs ) if not Alt_Model else ALT_MULTIMODEL( inputs ) # (T,|𝓘|,|A|) both cases
		return __MultiAdvArray( __tolist( outputs ), inputs.T, inputs.nI, inputs.nA ) # (T,|𝓘|,|A|)

# Get formatted inputs ⟶ run them through [ALT_]MULTIMODEL ⟶ format GPU tensor output into cpu array ⟶ return
cdef flt3 __MultiAdvEstimator( uint actingPlayer, infoset I, uint GPUrank=0, bint Alt_Model=FALSE ): #noexcept:
	
	cdef MMInputs inputs = MMInputs( actingPlayer, I, MULTIMODEL.IterSpan, GPUrank )

	with nn_eval_mode():

		if Alt_Model:
			return ALT_MULTIMODEL( inputs ).detach().to( device='cpu',dtype=tf32 ).numpy() # (T,|𝓘|,|A|)

		return MULTIMODEL( inputs ).detach().to( device='cpu',dtype=tf32 ).numpy() # (T,|𝓘|,|A|)

# Returns ∑{a}( α(I,a) ) ∀ I ∈ 𝓘
cdef flt1 __MultiAdvSums( flt2 posMultiAdvs ): #noexcept:

	cdef uint nI = posMultiAdvs.shape[0], I
	cdef flt1 multiSum = cyarr( (nI,), FLTSIZE, 'f' )

	for I from 0 <= I < nI: 
		multiSum[ I ] = ArrSum1d( posMultiAdvs[ I ] )

	return multiSum

# Does the same as __ActionProbs but for multiple I in parallel; multiAdvs shape = (|𝓘|,|A|)
cdef flt2 __MultiActionProbs( flt2 multiAdvs ): #noexcept: 
	
	# TODO: Keep this short-circuit?
	#if multiAdvs.shape[1]==1: 
		#return np.ones_like( multiAdvs ) # |A|=1 ⇒ σ(I,a) = 1 for the only a∈A
	
	cdef:
		uint  A=1
		flt2  aPlus          = NP( multiAdvs ).clip( min=0 )         # (|𝓘|,|A|)
		flt2  advSums        = NP( aPlus ).sum( axis=A,keepdims=1 )  # (|𝓘|,1)
		uint2 I_Has_Pos_Advs = Has_Nonzero( multiAdvs )              # (|𝓘|,1); =1 if ∃a∈A(I):α(I,a)>0 else 0

	if NP( I_Has_Pos_Advs ).all():
		return ArrDiv2d( aPlus, advSums ) # (|𝓘|,|A|); ∀I∈𝓘, ∃a∈A(I):α(I,a)>0


	UnzeroAdvs( advSums ) # turn 0s to 1s to avoid 0-div errors
	cdef:
		flt2 maxIAdvs = NP( multiAdvs ).max( axis=A, keepdims=1 ) # (|𝓘|,1);   max( α(I,A) ) ∀ I∈𝓘 
		flt2 Amax     = ArrEq2d( multiAdvs, maxIAdvs )            # (|𝓘|,|A|); {a∈A(I) | α(I,a)=max( α(I,A) )} ∀ I∈𝓘
		flt2 nAmax    = NumNonzero( Amax, along_axis=A )          # (|𝓘|,1);   |Amax(I)| ∀ I∈𝓘 
		flt2 posProbs = ArrDiv2d( aPlus, advSums )                # (|𝓘|,|A|); see below
		flt2 negProbs = ArrDiv2d( Amax, nAmax )                   # (|𝓘|,|A|); see below

	# ∀ I∈𝓘:  [ ∃a∈A(I):α(I,a)>0 ] ⇒ [ σ(I,a)=(α+)/Σ(α+) ∀a∈A(I) ]
	#         [ α(I,a)≤0 ∀a∈A(I) ] ⇒ [ σ(I,a)=1/|Amax| ∀a:α(I,a)=αmax, else 0 ]
	return np.where( I_Has_Pos_Advs, posProbs, negProbs ) # (|𝓘|,|A|)

# Estimate αᵗ(𝓘) ⟶ derive σᵗ(𝓘) ∀ t<T in parallel as above. Used during collection to init GTNode cgraphs
# TODO: Def need to write tests for this. Need to be absolutely certain indexing stuff here is correct
cdef flt3   MultiStrats( uint actingPlayer, infoset I, uint GPUrank=0, bint Alt_Model=FALSE, bint Legacy_Model=FALSE ): #noexcept:

	cdef:
		# Is 𝓘 one definite POV Infoset or many possible opp Infosets?
		bint  For_Opp = Are_Opponents( actingPlayer, I.POVplayer ),                                                    \
			  For_Pov = not For_Opp
		flt3  tAdvs   = ( __MultiAdvEstimator( actingPlayer, I, GPUrank, Alt_Model ) if not Legacy_Model else
		                  __LegacyMultiAdvEstimator( actingPlayer, I, GPUrank, Alt_Model ) ) # (T,|𝓘|,|A|) both cases
		uint  T       = tAdvs.shape[ 0 ],                                                                              \
			  nA      = tAdvs.shape[ 2 ],                                                                              \
			  nH      = NUM_POSSIBLE_HANDS,                                                                            \
			  t
		flt3  tStrats = NP( cyarr( (T,nH,nA), FLTSIZE, 'f' ),dtype=f32 )
		uint1 hInds

	if For_Opp:
		tStrats[:]=0
		hInds = NP( I.PossibleOppHandInds(),dtype=uintc )

	for t from 0 <= t < T:
		if For_Pov:
			tStrats[ t ][...] = __ActionProbs( tAdvs[ t ][ 0 ] )
		if For_Opp: # Insert σₒᵗ(Iₒ,ₕ) at h index ∀ opp hand h
			tStrats.base[ t ][ hInds.base ] = __MultiActionProbs( tAdvs[ t ] ) # .base is for np indexing magic

	return CONTIG( swapaxes( tStrats,0,2 ) ) # (|A|,1326,T); just useful downstream to have action axis first

# Actually returns an array where σᵃᵛᵍ(I,a) resides at [a,0], and σₜ(I,a) resides at [a,t] ∀ t∈[1,T]
cdef dbl2   AvgStrategy( infoset I, dbl1 iterReaches ): #noexcept:
	
	cdef:
		uint   T            = <uint>MULTIMODEL.IterSpan, ITER_AXIS=1
		dbl1   Trange       = np.arange( T,dtype='double' )
		dbl2   reachTerms   = newaxis( NP( Trange ) * NP( iterReaches ),axis=0 ) # (1,T)
		double totLinReach  = NP( reachTerms ).sum()

		dbl2  allIterStrats = NP( MultiStrats( I.POVplayer, I )[:,0,1:],dtype='double' ) # (|A|,T)
		dbl2  profileTerms  = NP( reachTerms ) * NP( allIterStrats )   # (|A|,T)
		dbl1  cumulProfile  = NP( profileTerms ).sum( axis=ITER_AXIS ) # (|A|,)

		uint  nA
		dbl1  avgStrat
		dbl2  stratArray

	avgStrat   = NP( cumulProfile ) / totLinReach
	nA         = avgStrat.shape[ 0 ]
	stratArray = cyarr( (nA,T+1), DBLSIZE, 'd' ) 

	stratArray[:,0 ] = avgStrat
	stratArray[:,1:] = allIterStrats
	return stratArray # (|A|,T+1)


# *-* # 