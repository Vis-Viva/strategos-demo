# distutils: language = c
# cython: language_level 3
# cython: profile = False


from libc.stdlib cimport malloc, realloc, free

cimport cython
cimport numpy as cnp
cnp.import_array()

from cpython.array cimport clone as pyarr
from cython.view   cimport array as cyarr

cimport strategos_tools.utils.funcs as util
from strategos_tools.core.CONSTS      cimport *
from strategos_tools.env.player_ops   cimport Are_Opponents
from strategos_tools.env.gamenode_ops cimport DummyNode

import datetime, pickle, numpy as np
from random import shuffle as permute
from os     import listdir, remove as destroy, getcwd as cwd
from time   import time as TimeNow

import torch as pt
from numpy import asarray as NP, ascontiguousarray as CONTIG, float32 as f32, float64 as f64, uintc, intc
from torch import as_tensor as TENSOR, float32 as tf32, int32 as tintc
#from torch import from_numpy as TENSOR


# ==================================================================================================
# This module provides some very fast containers and transform utilities which make up strategos's
# data-structuring backbone, as well as some record-keeping tools. These make up the translation
# layer between game trajectory data and NN data for multiple model types (legacy and modern,
# different architectures and input structuring). The priority here is memory management to meet 
# very strict array alignment/contiguity reqs and ensure copyless movement of data.
# ==================================================================================================


# Generates and stores single-iter AdvNet inference input tensors from an infoset
cdef class AdvNetInputs:

	def __init__( self, infoset I, uint GPUrank=0 ): 
		self.__INIT__( I, GPUrank )

	cdef void  __INIT__( self, infoset I, uint GPUrank=0 ): #noexcept:

		self.GPU = f"cuda:{GPUrank}"
		self.__init_history( I )
		self.__init_cards( I )
		self.__init_actions( I )

	# Constructs game history tensor
	cdef void  __init_history( self, infoset I ): #noexcept:

		cdef uint3 hArr = cyarr( (1,I.hLen,EVEC_SIZE), UINTSIZE, 'I' )
		hArr[0] = I.ObservableHistory()
		self.H  = pt.tensor( NP( hArr,dtype=f32 ), device=self.GPU )

	# Constructs tensors for observable cards
	cdef void  __init_cards( self, infoset I ): #noexcept:

		cdef:
			uint3 hcArr  = cyarr( (1, MAX_HOLE_CARDS, CVEC_SIZE), UINTSIZE, 'I' ),                                     \
				  fcArr  = cyarr( (1, FLOP_DEAL_SIZE, CVEC_SIZE), UINTSIZE, 'I' )
			uint2 tcArr  = cyarr( (1, CVEC_SIZE), UINTSIZE, 'I' ),                                                     \
				  rcArr  = cyarr( (1, CVEC_SIZE), UINTSIZE, 'I' ),                                                     \
				  hCards = I.HoleCards( Fill_To_Max=TRUE ),                                                            \
				  bCards = I.BoardCards( Fill_To_Max=TRUE )

		hcArr[ 0 ] = hCards
		fcArr[ 0 ] = bCards[:3 ] # first three bcards are flop deal
		tcArr[ 0 ] = bCards[ 3 ] # fourth is turn deal
		rcArr[ 0 ] = bCards[ 4 ] # fifth is river deal

		self.hC_c = pt.tensor( NP( hcArr[ :,:,CARD ], dtype=intc ), device=self.GPU ) # hole cardIDs
		self.hC_r = pt.tensor( NP( hcArr[ :,:,RANK ], dtype=intc ), device=self.GPU ) # hole rankIDs
		self.hC_s = pt.tensor( NP( hcArr[ :,:,SUIT ], dtype=intc ), device=self.GPU ) # hole suitIDs

		self.fC_c = pt.tensor( NP( fcArr[ :,:,CARD ], dtype=intc ), device=self.GPU ) # flop cardIDs
		self.fC_r = pt.tensor( NP( fcArr[ :,:,RANK ], dtype=intc ), device=self.GPU ) # flop rankIDs
		self.fC_s = pt.tensor( NP( fcArr[ :,:,SUIT ], dtype=intc ), device=self.GPU ) # flop suitIDs

		self.tC_c = pt.tensor( NP( tcArr[ :,CARD ],   dtype=intc ), device=self.GPU ) # turn cardIDs
		self.tC_r = pt.tensor( NP( tcArr[ :,RANK ],   dtype=intc ), device=self.GPU ) # turn rankIDs
		self.tC_s = pt.tensor( NP( tcArr[ :,SUIT ],   dtype=intc ), device=self.GPU ) # turn suitIDs

		self.rC_c = pt.tensor( NP( rcArr[ :,CARD ],   dtype=intc ), device=self.GPU ) # river cardIDs
		self.rC_r = pt.tensor( NP( rcArr[ :,RANK ],   dtype=intc ), device=self.GPU ) # river rankIDs
		self.rC_s = pt.tensor( NP( rcArr[ :,SUIT ],   dtype=intc ), device=self.GPU ) # river suitIDs

	# Constructs tensors for available actions
	cdef void  __init_actions( self, infoset I ): #noexcept:

		cdef uint2 aArr = actionset( I ).AMat()
		self.nA = aArr.shape[ 0 ]
		self.A  = pt.tensor( NP( aArr,dtype=f32 ), device=self.GPU )

	# Generates dummy AdvNet data, useful for when we need to do a .forward() call for compile
	@staticmethod
	cdef AdvNetInputs _DummyInputs( uint GPUrank=0 ): #noexcept:

		cdef gamenode dummyNode = DummyNode()
		cdef infoset  dumbI     = infoset( dummyNode, perspective_of=dummyNode.ActingPlayer() )
		return AdvNetInputs( dumbI,GPUrank )

	# Just lets us call the above from python NN code
	@staticmethod
	def DummyInputs( uint GPUrank=0 ):
		return AdvNetInputs._DummyInputs( GPUrank )


# Same deal as AdvNetInputs, but for MultiModel. Unlike AdvNet, MultiModel is used both in situations
# where we're doing inference on a single definite infoset for the POV player, and where we're doing
# it for a whole range of potential opponent infosets (each corresponding to a possible hand they 
# may have been dealt). Hence, MMInputs must handle both of these situations.
cdef class MMInputs:

	def __init__( self, uint actingPlayer, infoset Ipov, uint iterSpan, uint GPUrank=0 ): 
		self.__INIT__( actingPlayer, Ipov, iterSpan, GPUrank )

	cdef void  __INIT__( self, uint actingPlayer, infoset Ipov, uint iterSpan, uint GPUrank=0 ): #noexcept:

		self.T   = iterSpan + 1 # Useful downstream to store this on MMInputs
		self.GPU = f"cuda:{GPUrank}"

		# Is POV or OPP acting at the position we're evaluating? Tells us whose hole cards to use
		cdef bint Opp_State = Are_Opponents( actingPlayer, Ipov.POVplayer )
		
		# Infoset for the current acting player, which may or may not be the POV player
		cdef infoset Iap = Ipov if not Opp_State else infoset( sourceNode=Ipov._n, perspective_of=actingPlayer )

		self.__init_history( Ipov,Opp_State )
		self.__init_cards( Ipov,Opp_State )
		self.__init_actions( Iap )

		self.nSamples = self.nI * self.nA

	# Constructs history tensor, accounts for whether we're doing pov or opp inference
	cdef void __init_history( self, infoset Ipov, bint Opp_State ): #noexcept:

		cdef tuple hShape
		cdef uint3 hArr

		# many potential histories corresponding to possible opponent hands
		if Opp_State: 
			hArr = Ipov.PossibleOppHistories()

		# one definite known POV history
		else:
			hShape    = (1, Ipov.hLen, EVEC_SIZE)
			hArr      = cyarr( hShape, UINTSIZE, 'I' )
			hArr[ 0 ] = Ipov.ObservableHistory()

		self.nI = hArr.shape[ 0 ] # number of distinct infosets we're evaluating actions for
		self.H  = pt.tensor( NP( hArr,dtype=f32 ), device=self.GPU )

	# Constructs card tensors, accounts for whether we're doing pov or opp inference
	cdef void __init_cards( self, infoset Ipov, bint Opp_State ): #noexcept:

		# Initialize arrays we'll subsequently fill
		cdef:
			tuple hShape = (1, MAX_HOLE_CARDS, CVEC_SIZE)
			uint3 hcArr  = cyarr( hShape, UINTSIZE,'I' ) if not Opp_State else Ipov.PossibleOppHands(),                \
				  fcArr  = cyarr( (1, FLOP_DEAL_SIZE, CVEC_SIZE), UINTSIZE, 'I' )
			uint2 tcArr  = cyarr( (1, CVEC_SIZE), UINTSIZE, 'I' ),                                                     \
				  rcArr  = cyarr( (1, CVEC_SIZE), UINTSIZE, 'I' ),                                                     \
				  bCards = Ipov.BoardCards( Fill_To_Max=TRUE )

		# If Opp_State, we already got all possible opp hole cards above
		if not Opp_State: 
			hcArr[ 0 ] = Ipov.HoleCards( Fill_To_Max=TRUE )

		# Board cards are known regardless of Opp_State
		fcArr[ 0 ] = bCards[:3 ] # flop  = first three board deals
		tcArr[ 0 ] = bCards[ 3 ] # turn  = fourth board deal
		rcArr[ 0 ] = bCards[ 4 ] # river = fifth board deal

		self.hC_c = pt.tensor( NP( hcArr[ :,:,CARD ], dtype=intc ), device=self.GPU ) # hole cardIDs
		self.hC_r = pt.tensor( NP( hcArr[ :,:,RANK ], dtype=intc ), device=self.GPU ) # hole rankIDs
		self.hC_s = pt.tensor( NP( hcArr[ :,:,SUIT ], dtype=intc ), device=self.GPU ) # hole suitIDs

		self.fC_c = pt.tensor( NP( fcArr[ :,:,CARD ], dtype=intc ), device=self.GPU ) # flop cardIDs
		self.fC_r = pt.tensor( NP( fcArr[ :,:,RANK ], dtype=intc ), device=self.GPU ) # flop rankIDs
		self.fC_s = pt.tensor( NP( fcArr[ :,:,SUIT ], dtype=intc ), device=self.GPU ) # flop suitIDs

		self.tC_c = pt.tensor( NP( tcArr[ :,CARD ],   dtype=intc ), device=self.GPU ) # turn cardIDs
		self.tC_r = pt.tensor( NP( tcArr[ :,RANK ],   dtype=intc ), device=self.GPU ) # turn rankIDs
		self.tC_s = pt.tensor( NP( tcArr[ :,SUIT ],   dtype=intc ), device=self.GPU ) # turn suitIDs

		self.rC_c = pt.tensor( NP( rcArr[ :,CARD ],   dtype=intc ), device=self.GPU ) # river cardIDs
		self.rC_r = pt.tensor( NP( rcArr[ :,RANK ],   dtype=intc ), device=self.GPU ) # river rankIDs
		self.rC_s = pt.tensor( NP( rcArr[ :,SUIT ],   dtype=intc ), device=self.GPU ) # river suitIDs

	# Constructs actionset tensor from an actionset matrix
	cdef void __init_actions( self, infoset Iap ): #noexcept:

		cdef uint2 aArr = actionset( Iap ).AMat()
		self.nA = aArr.shape[ 0 ]
		self.A  = pt.tensor( NP( aArr,dtype=f32 ), device=self.GPU )

	# Generates some dummy MM input data, useful for when we need to do a .forward() call for compile
	@staticmethod
	cdef MMInputs _DummyInputs( uint iterSpan, uint GPUrank=0 ): #noexcept:

		cdef:
			gamenode dummyNode  = DummyNode()
			uint     dumbPlayer = dummyNode.ActingPlayer()
			infoset  dumbI      = infoset( dummyNode, perspective_of=dumbPlayer )

		return MMInputs( dumbPlayer, dumbI, iterSpan, GPUrank )

	# Just lets us call the above from python NN code
	@staticmethod
	def DummyInputs( uint iterSpan, uint GPUrank=0 ): 
		return MMInputs._DummyInputs( iterSpan,GPUrank )


# Legacy MM input format for doing evals against old non-transformer models w old card vector format
cdef class MMInputs_old:

	def __init__( self, uint actingPlayer, infoset Ipov, uint iterSpan, uint GPUrank=0 ): 
		self.__INIT__( actingPlayer, Ipov, iterSpan, GPUrank )

	cdef void  __INIT__( self, uint actingPlayer, infoset Ipov, uint iterSpan, uint GPUrank=0 ): #noexcept:

		self.T = iterSpan + 1
		self.OLD_CVEC_SIZE = 17
		self.OLD_EVEC_SIZE = 24
		self.NEW_EVEC_SIZE = EVEC_SIZE
		self.HOLEVEC_SIZE  = self.OLD_CVEC_SIZE * MAX_HOLE_CARDS  # = 34
		self.BOARDVEC_SIZE = self.OLD_CVEC_SIZE * MAX_BOARD_CARDS # = 85

		if Are_Opponents( actingPlayer,Ipov.POVplayer ): 
			self.__init_opp_state( Ipov,GPUrank )
		else:
			self.__init_pov_state( Ipov,GPUrank )

	# Orchestrates construction of legacy inputs for POV player states
	cdef void  __init_pov_state( self, infoset Ipov, uint GPUrank ): #noexcept:

		cdef:
			uint1   dealSteps  = Ipov.DealSteps()
			uint[:] hCardIDs   = Ipov.HoleCards( Fill_To_Max=TRUE )[ :,CARD ],                                         \
				    bCardIDs   = Ipov.BoardCards( Fill_To_Max=TRUE )[ :,CARD ] # not contiguous, so can't be uint1
			uint2   convHCards = cyarr( (1,self.HOLEVEC_SIZE),  UINTSIZE, 'I' ),                                       \
				    convBCards = cyarr( (1,self.BOARDVEC_SIZE), UINTSIZE, 'I' ),                                       \
				    convASet   = self.__ActionsetConverter( actionset( Ipov ).AMat() ),                                \
				    hcArr      = self.__CardConverter( hCardIDs ),                                                     \
				    bcArr      = self.__CardConverter( bCardIDs )
			uint3   hArr       = cyarr( (1, Ipov.hLen, self.NEW_EVEC_SIZE), UINTSIZE, 'I' ), convHist
			uint    nA         = convASet.shape[ 0 ]
		hArr[ 0 ] = Ipov.ObservableHistory()

		convHist      = self.__HistoryConverter( hArr,dealSteps )
		convHCards[0] = self.__MultiCardVec( hcArr )
		convBCards[0] = self.__MultiCardVec( bcArr )

		self.nI = 1
		self.nA = nA
		self.nSamples = nA # Only running inference on 1 definite POV infoset I, so total num samples is just |A(I)|
		self.__init_tensors( convHist, convHCards, convBCards, convASet, GPUrank )

	# Orchestrates construction of legacy inputs for opponent states
	cdef void  __init_opp_state( self, infoset Ipov, uint GPUrank ): #noexcept:

		cdef:
			infoset   Iopp       = infoset( sourceNode=Ipov._n, perspective_of=Ipov.OPPplayer )
			uint1     dealSteps  = Ipov.DealSteps()
			uint[:]	  bCardIDs   = Ipov.BoardCards( Fill_To_Max=TRUE )[ :,CARD ] # uncontig, can't be uint1
			uint2     convASet   = self.__ActionsetConverter( actionset(Iopp).AMat() )
			uint[:,:] hCardIDs   = Ipov.PossibleOppHands()[ :,:,CARD ] # not contig, can't be uint2
			uint      nH         = hCardIDs.shape[ 0 ],                                                                \
					  nA         = convASet.shape[ 0 ],                                                                \
					  h
			uint2     convHCards = cyarr( (nH,self.HOLEVEC_SIZE), UINTSIZE, 'I' ),                                     \
					  convBCards = cyarr( (1,self.BOARDVEC_SIZE), UINTSIZE, 'I' ),                                     \
					  bcArr      = self.__CardConverter( bCardIDs ),                                                   \
					  hcArr
			uint3     convHist   = self.__HistoryConverter( Ipov.PossibleOppHistories(),dealSteps )

		convBCards[0] = self.__MultiCardVec( bcArr )

		for h from 0 <= h < nH:
			hcArr         = self.__CardConverter( hCardIDs[h] )
			convHCards[h] = self.__MultiCardVec( hcArr )

		self.nI = nH
		self.nA = nA
		self.nSamples = nH*nA # Each action has to be eval'd against each possible opponent infoset
		self.__init_tensors( convHist, convHCards, convBCards, convASet, GPUrank )

	cdef void  __init_tensors( self, uint3 H, uint2 hC, uint2 bC, uint2 A, uint GPUrank ): #noexcept:
		cdef str GPU = f"cuda:{GPUrank}"
		self.H  = pt.tensor( NP( H, dtype=f32 ), device=GPU )
		self.hC = pt.tensor( NP( hC,dtype=f32 ), device=GPU )
		self.bC = pt.tensor( NP( bC,dtype=f32 ), device=GPU )
		self.A  = pt.tensor( NP( A, dtype=f32 ), device=GPU )

	# Converts modern single-int card IDs to old binary vector representation
	cdef uint2 __CardConverter( self, uint[:] cardIDs ): #noexcept:

		cdef:
			uint  nCards   = cardIDs.shape[ 0 ], rankIdx, suitIdx, c, cID
			uint2 cardVecs = cyarr( (nCards, self.OLD_CVEC_SIZE), UINTSIZE, 'I' )

		# Set cardVecs to all zeros, then insert 1s at appropriate rank & suit indices
		cardVecs[:]=0
		for c from 0 <= c < nCards:
			cID = cardIDs[ c ]

			if cID != 0:
				rankIdx = FULL_VEC_DECK[ cID,RANK ]-1
				suitIdx = FULL_VEC_DECK[ cID,SUIT ]-1 + NUM_RANKS
				cardVecs[ c,rankIdx ] = 1
				cardVecs[ c,suitIdx ] = 1

		return cardVecs

	# Reshape hArr (nH,hLen,8) ⟶ (nH,hLen,24), fill extra space w old cardVecs derived from new cardIDs
	cdef uint3 __HistoryConverter( self, uint3 hArr, uint1 dealSteps ): #noexcept:

		cdef:
			uint    nH = hArr.shape[ 0 ], hLen = hArr.shape[ 1 ], nDeals = dealSteps.shape[ 0 ], d, dStep
			uint3   convertedHist = cyarr( (nH, hLen, self.OLD_EVEC_SIZE), UINTSIZE, 'I' )
			uint[:] stepCard # uint[:] instead of uint1 because what we assign here isn't contiguous

		# Make room for old card vecs to replace card ints
		convertedHist[ :,:,:self.NEW_EVEC_SIZE ] = hArr
		convertedHist[ :,:,self.NEW_EVEC_SIZE: ] = 0

		for d from 0 <= d < nDeals:
			dStep = dealSteps[ d ]

			if dStep > 0: # dStep==0 implies card d hasn't been dealt yet
				stepCard = hArr[ :,dStep,CDEALT ] # card dealt @ this step across all input histories
				convertedHist[ :,dStep,CDEALT: ] = self.__CardConverter( stepCard )

		return convertedHist

	# Takes array of multiple old card vecs and concats them; replicates legacy multi-card format
	cdef uint1 __MultiCardVec( self, uint2 cardVecs ): #noexcept:

		cdef uint  nCards   = cardVecs.shape[ 0 ], c, start, stop
		cdef uint1 multiVec = pyarr( ARR_TMPLT_I, nCards*self.OLD_CVEC_SIZE, zero=False )

		for c from 0 <= c < nCards:
			start = c * self.OLD_CVEC_SIZE
			stop  = (c+1) * self.OLD_CVEC_SIZE
			multiVec[ start:stop ] = cardVecs[ c ]

		return multiVec

	# Expands size of event vectors in an actionset matrix for input compat with legacy models
	cdef uint2 __ActionsetConverter( self, uint2 A ): #noexcept:

		cdef uint  nA = A.shape[ 0 ]
		cdef uint2 convertedA = cyarr( (nA,self.OLD_EVEC_SIZE), UINTSIZE, 'I' )

		convertedA[ :,EVEC_SIZE: ] = 0
		convertedA[ :,:EVEC_SIZE ] = A

		return convertedA


# A single AdvNet input/output sample. Inputs: (I, aVec). Output target: α(I,a)
cdef class advsample:

	def __init__( self, uint iterNum, infoset I, uint1 actionVec, float actionAdv ):
		self.__INIT__( iterNum, I, actionVec, actionAdv )

	cdef void    __INIT__( self, uint iterNum, infoset I, uint1 actionVec, float actionAdv ): #noexcept:
		self.t       = iterNum
		self.Infoset = I
		self.aVec    = actionVec
		self.aAdv    = actionAdv

	# Outputs a pydict that can be saved to disk then used to reconstruct the original advsample on load
	cdef dict      to_dict( self ): #noexcept:
		return { 't':     self.t,
				 'Idict': self.Infoset.to_dict(),
				 'aVec':  NP( self.aVec,dtype=uintc ),
				 'aAdv':  self.aAdv }

	cdef void      save( self, str to_file ): #noexcept:
		with open( to_file,'ab' ) as advFile: 
			pickle.dump( self.to_dict(), advFile, protocol=-1 )

	# Secondary constructor - inverse of to_dict
	@staticmethod
	cdef advsample from_dict( dict advDict ): #noexcept:
		cdef infoset I = infoset.from_dict( advDict[ 'Idict' ] )
		return advsample( iterNum=advDict[ 't' ], I=I, actionVec=advDict[ 'aVec' ], actionAdv=advDict[ 'aAdv' ] )


# Maps an infoset I to α(I,a) targets ∀ a∈A(I) - lets us efficiently construct many advsamples from one I
cdef class advmap:

	def __init__( self, int for_iter, infoset I, int1 aInds, flt1 aTargets ):
		self.__INIT__( for_iter, I, aInds, aTargets )

	cdef void __INIT__( self, int for_iter, infoset I, int1 aInds, flt1 aTargets ): #noexcept:
		self.IterNum    = for_iter
		self.NumSamples = aTargets.shape[ 0 ] 
		self.Infoset    = I
		self.ActionInds = list( aInds )
		self.AdvTargets = list( aTargets )

	# Human-readable advmap metadata printout
	cdef void   summary( self ): #noexcept:
		print( '\n'+('='*25) )
		print( "ADVMAP SUMMARY".center(25) )
		print( ('='*25)+'\n' )
		print( f"\tIterNum     = {self.IterNum}" )
		print( f"\tNumSamples  = {self.NumSamples}" )
		print( f"\tAction Inds = {self.ActionInds}" )
		print( f"\tAdvTargets  = {self.AdvTargets}" )

	# Outputs a pydict that can be saved to disk then used to reconstruct the original advmap on load
	cdef dict   to_dict( self ): #noexcept:

		cdef dict d = {}
		d[ 'IterNum' ]    = self.IterNum
		d[ 'NumSamples' ] = self.NumSamples
		d[ 'ActionInds' ] = self.ActionInds
		d[ 'AdvTargets' ] = self.AdvTargets
		d[ 'Idict' ]      = self.Infoset.to_dict()
		return d

	# DEPRECATED: We don't actually save advmaps anymore, we just save individual samples
	cdef void   save( self, str to_file ): #noexcept:
		cdef dict advDict = self.to_dict()
		with open( to_file,'ab' ) as advFile: 
			pickle.dump( advDict, advFile, protocol=-1 )

	# Serialization helper: ∀ sample s ∈ advmap, do equiv of s.to_dict(), return list of all dicts
	cdef list __extract_sample_dicts( self ): #noexcept:

		cdef:
			uint      nA      = <uint>len( self.ActionInds ), t = self.IterNum, a, aIdx
			infoset   I       = self.Infoset
			uint2     A       = actionset( I ).AMat()
			dict      Idict   = I.to_dict(), sDict
			list      Isamples=[]
			uint1     aVec
			float     aAdv

		for a from 1 <= a <= nA:
			aIdx  = self.ActionInds[ a-1 ]
			aVec  = A[ aIdx ]
			aAdv  = self.AdvTargets[ a-1 ]
			sDict = {'t': t, 'Idict': Idict, 'aVec': np.array( aVec,dtype=uintc ), 'aAdv': aAdv }
			Isamples.append( sDict )

		return Isamples

	# Preferred advmap serialization; instead of the advmap itself, just serialize all its samples
	cdef void   save_sample_dicts( self, str to_file ): #noexcept:

		cdef list sDicts = self.__extract_sample_dicts()
		cdef dict sDict

		with open( to_file,'ab' ) as sampleFile:
			for sDict in sDicts:
				pickle.dump( sDict, sampleFile, protocol=-1 )

	# Secondary constructor - inverse of to_dict
	@staticmethod
	cdef advmap from_dict( dict advdict ): #noexcept:

		cdef:
			list    _aInds   = advdict[ 'ActionInds' ],                                                                \
					_aAdvs   = advdict[ 'AdvTargets' ]
			int     t        = <int>advdict['IterNum'],                                                                \
					nSamples = <int>advdict['NumSamples'],                                                             \
					nA       = <int>len( _aInds ), a
			infoset I        = infoset.from_dict( advdict[ 'Idict' ] )
			int1    aInds    = cyarr( (nA,), INTSIZE, 'i' )
			flt1    aAdvs    = cyarr( (nA,), FLTSIZE, 'f' )

		for a from 0 <= a < nA:
			aInds[ a ] = _aInds[ a ]
			aAdvs[ a ] = _aAdvs[ a ]

		return advmap( t, I, aInds, aAdvs )


	# ----- PYTHON INTERFACE ---------------------------------------------------
	# Just some stuff we need at the pure-python level (i.e. training scripts)
	
	def extract_samples( self ):
		return self._extract_samples()

	def num_targets( self ):
		return self.NumSamples


# Batched GPU-allocated AdvNet training samples, very specific highly efficient memory layout.
# Includes fwd-pass inputs and advantage output targets.
cdef class DataBatch:

	# TODO: Roll these inputs up into a tuple or something, this function signature is a warcrime.
	def __init__( self, int3 H,
						uint2 hCc, uint2 hCr, uint2 hCs,
						uint2 fCc, uint2 fCr, uint2 fCs,
						uint1 tCc, uint1 tCr, uint1 tCs,
						uint1 rCc, uint1 rCr, uint1 rCs,
						uint2 A,   flt1  V,   uint1 W,   uint2 M, str GPU ):

		self.__INIT__( H, hCc,hCr,hCs, fCc,fCr,fCs, tCc,tCr,tCs, rCc,rCr,rCs, A, V, W, M, GPU )


	# TODO: Roll these inputs up into a tuple or something, this function signature is a warcrime.
	cdef void __INIT__( self, int3 H,
							  uint2 hCc, uint2 hCr, uint2 hCs,
							  uint2 fCc, uint2 fCr, uint2 fCs,
							  uint1 tCc, uint1 tCr, uint1 tCs,
							  uint1 rCc, uint1 rCr, uint1 rCs,
							  uint2 A,   flt1 V,    uint1 W,   uint2 M, str GPU ): #noexcept:

		self.size = H.shape[ 0 ] # Each sample has one history, so batch size = num histories
		self.H    = TENSOR( NP( H,  dtype=f32  ), device=GPU ) # Game histories
		self.hCc  = TENSOR( NP( hCc,dtype=intc ), device=GPU ) # Hole cardIDs
		self.hCr  = TENSOR( NP( hCr,dtype=intc ), device=GPU ) # Hole rankIDs
		self.hCs  = TENSOR( NP( hCs,dtype=intc ), device=GPU ) # Hole suitIDs
		self.fCc  = TENSOR( NP( fCc,dtype=intc ), device=GPU ) # Flop cardIDs
		self.fCr  = TENSOR( NP( fCr,dtype=intc ), device=GPU ) # Flop rankIDs
		self.fCs  = TENSOR( NP( fCs,dtype=intc ), device=GPU ) # Flop suitIDs
		self.tCc  = TENSOR( NP( tCc,dtype=intc ), device=GPU ) # Turn cardIDs
		self.tCr  = TENSOR( NP( tCr,dtype=intc ), device=GPU ) # Turn rankIDs
		self.tCs  = TENSOR( NP( tCs,dtype=intc ), device=GPU ) # Turn suitIDs
		self.rCc  = TENSOR( NP( rCc,dtype=intc ), device=GPU ) # River cardIDs
		self.rCr  = TENSOR( NP( rCr,dtype=intc ), device=GPU ) # River rankIDs
		self.rCs  = TENSOR( NP( rCs,dtype=intc ), device=GPU ) # River suitIDs
		self.A    = TENSOR( NP( A,  dtype=f32  ), device=GPU ) # Action vectors
		self.V    = TENSOR( NP( V,  dtype=f32  ), device=GPU ) # Adv targets
		self.W    = TENSOR( NP( W,  dtype=f32  ), device=GPU ) # Samples weighted with iterNum they were collected on
		self.M    = TENSOR( NP( M,  dtype=bool ), device=GPU ) # Bool transformer mask for nonuniform history inputs


# Specialized replacement for PyTorch's bloated DataLoader. 
# Allocates, populates, and exposes batched training data in a way compatible with multi-GPU training.
cdef class DATAMACHINE:

	def __init__( self, list shuffledSamples, uint bsize, int world_size, int rank ):
		self.__INIT__( shuffledSamples, bsize, world_size, rank )

	# Determine mem layout from sample shapes ⟶ allocate mem ⟶ fill allocated storage ⟶ partition into batches
	cdef void     __INIT__( self, list shuffledSamples, uint bsize, int world_size, int rank ): #noexcept:

		self.WORLD_SIZE = world_size
		self.RANK       = rank
		self.GPU        = f"cuda:{rank}"
		self.__determine_storage_layout( shuffledSamples, bsize )
		self.__allocate_temp_storage()
		self.__populate_temp_storage( shuffledSamples )
		self.__partition_storage()
		self.__destroy_temp_storage() # temp storage no longer needed after partitioning

	# Need to know max(|H|) to allocate a contiguous array for all histories. Faster than python max()
	cdef void     __find_max_hLen( self, list from_samples ): #noexcept:

		cdef:
			advsample s
			uint      nSamples = <uint>len( from_samples ), L
			list      hLengths = [ s.Infoset.hLen for s in from_samples ]

		self.Lmax = 0
		for L in hLengths:
			if L > self.Lmax: 
				self.Lmax = L

	# Calculates storage partitioning boundaries etc. shuffledSamples = WHOLE global collected dataset
	cdef void     __determine_storage_layout( self, list shuffledSamples, uint bsize ): #noexcept:

		self.__find_max_hLen( from_samples=shuffledSamples )

		cdef uint S = <uint>len( shuffledSamples ), subsetSize = S//self.WORLD_SIZE, s, hLen

		self.nSamples         = subsetSize
		self.BatchSize        = bsize if bsize < subsetSize else subsetSize
		self.nBatches         = (self.nSamples//self.BatchSize)
		self._dataStart       = self.RANK * subsetSize          # Global dataset idx for start of this DM's data slice
		self._dataStop        = self._dataStart + self.nSamples # Corresponding global end idx
		self._final_bsize     = self.nSamples % self.BatchSize  # Account for possible irregular-size trailing batch
		self._batches_uniform = self._final_bsize <= 1          # <=1 (not ==0) since BNorm1d explodes on bsize 1 lol
		if not self._batches_uniform: self.nBatches+=1          # To account for irregular final batch if present

	# Allocates mem space for temp storage. Temp storage holds DM's full data subset before batching
	cdef void     __allocate_temp_storage( self ): #noexcept:

		cdef tuple Hshape  = ( self.nSamples, self.Lmax, EVEC_SIZE ),                                                  \
				   hCshape = ( self.nSamples, MAX_HOLE_CARDS ),                                                        \
				   fCshape = ( self.nSamples, FLOP_DEAL_SIZE ),                                                        \
				   tCshape = ( self.nSamples, ),                                                                       \
				   rCshape = ( self.nSamples, ),                                                                       \
				   Ashape  = ( self.nSamples, EVEC_SIZE ),                                                             \
				   Vshape  = ( self.nSamples, ),                                                                       \
				   Wshape  = ( self.nSamples, ),                                                                       \
				   Mshape  = ( self.nSamples, self.Lmax )

		self._H    = cyarr( Hshape,  INTSIZE,  'i' ) # histories
		self._H[:] = -1                              # padding for variable hLen
		self._hCc  = cyarr( hCshape, UINTSIZE, 'I' ) # hole cardIDs
		self._hCr  = cyarr( hCshape, UINTSIZE, 'I' ) # hole rankIDs
		self._hCs  = cyarr( hCshape, UINTSIZE, 'I' ) # hole suitIDs
		self._fCc  = cyarr( fCshape, UINTSIZE, 'I' ) # flop cardIDs
		self._fCr  = cyarr( fCshape, UINTSIZE, 'I' ) # flop rankIDs
		self._fCs  = cyarr( fCshape, UINTSIZE, 'I' ) # flop suitIDs
		self._tCc  = cyarr( tCshape, UINTSIZE, 'I' ) # turn cardIDs
		self._tCr  = cyarr( tCshape, UINTSIZE, 'I' ) # turn rankIDs
		self._tCs  = cyarr( tCshape, UINTSIZE, 'I' ) # turn suitIDs
		self._rCc  = cyarr( rCshape, UINTSIZE, 'I' ) # river cardIDs
		self._rCr  = cyarr( rCshape, UINTSIZE, 'I' ) # river rankIDs
		self._rCs  = cyarr( rCshape, UINTSIZE, 'I' ) # river suitIDs
		self._A    = cyarr( Ashape,  UINTSIZE, 'I' ) # actions
		self._V    = cyarr( Vshape,  FLTSIZE,  'f' ) # advantage targets
		self._W    = cyarr( Wshape,  UINTSIZE, 'I' ) # sample weights
		self._M    = cyarr( Mshape,  UINTSIZE, 'I' ) # history transformer mask for nonuniform hLen

	# Generates a binary transformer mask for a single sample history
	cdef uint1    __history_mask( self, uint hLen ): #noexcept:

		cdef uint  i
		cdef uint1 sampleMask = cyarr( (self.Lmax,), UINTSIZE, 'I' )

		for i from 0 <= i < self.Lmax: 
			sampleMask[ i ] = <uint>(i >= hLen) # god zero indexing is just terrible
			
		return sampleMask

	# Get sample at global idx s ⟶ extract data from it ⟶ copy data into temp storage at local idx
	cdef void     __populate_temp_storage( self, list shuffledSamples ): #noexcept:

		cdef: 
			advsample sample
			infoset   I
			int2      iHist
			uint2     iHole, iBoard, iFlop
			uint1     iTurn, iRiver, iMask
			uint1     aVec
			float     aAdv
			uint      s,h,i,t

		# Take adv sample at global index s and populate local DATAMACHINE index i with it
		for s from self._dataStart <= s < self._dataStop:
			sample = shuffledSamples[ s ]
			t      = sample.t       # Sample weight = CFR iter it was collected on
			aVec   = sample.aVec    # Sample action vector
			aAdv   = sample.aAdv    # Sample adv target
			I      = sample.Infoset # Source of sample history & cards
			h      = I.hLen
			iHole  = I.HoleCards( Fill_To_Max=TRUE )  # Sample hole cards
			iBoard = I.BoardCards( Fill_To_Max=TRUE ) # Sample board cards
			iFlop  = iBoard[:3 ]                      # Sample flop cards
			iTurn  = iBoard[ 3 ]                      # Sample turn cards
			iRiver = iBoard[ 4 ]                      # Sample river cards
			iHist  = NP( I.ObservableHistory(),dtype=intc ) # Gotta NP this to change uint ⟶ int (ugh)
			iMask  = self.__history_mask( h )

			i = s - self._dataStart # Local storage index to copy data into
			
			self._H[ i ][ :h ] = iHist # Sample s game history
			
			self._hCc[ i ] = iHole[:,CARD ] # Sample s hole cardIDs
			self._hCr[ i ] = iHole[:,RANK ] # Sample s hole rankIDs
			self._hCs[ i ] = iHole[:,SUIT ] # Sample s hole suitIDs
			
			self._fCc[ i ] = iFlop[:,CARD ] # Sample s flop cardIDs
			self._fCr[ i ] = iFlop[:,RANK ] # Sample s flop rankIDs
			self._fCs[ i ] = iFlop[:,SUIT ] # Sample s flop suitIDs
			
			self._tCc[ i ] = iTurn [ CARD ] # Sample s turn cardIDs
			self._tCr[ i ] = iTurn [ RANK ] # Sample s turn rankIDs
			self._tCs[ i ] = iTurn [ SUIT ] # Sample s turn suitIDs
			
			self._rCc[ i ] = iRiver[ CARD ] # Sample s river cardIDs
			self._rCr[ i ] = iRiver[ RANK ] # Sample s river rankIDs
			self._rCs[ i ] = iRiver[ SUIT ] # Sample s river suitIDs

			self._A[ i ] = aVec  # Sample s action vectors
			self._V[ i ] = aAdv  # Sample s advantage targets
			self._W[ i ] = t     # Sample s weight
			self._M[ i ] = iMask # Sample s history mask

	# No need for all this unbatched stuff sitting in mem receiving tax dollars if it's no longer doing work for us
	cdef void     __destroy_temp_storage( self ): #noexcept:
		self._H   = None
		self._hCc = None
		self._hCr = None
		self._hCs = None
		self._fCc = None
		self._fCr = None
		self._fCs = None
		self._tCc = None
		self._tCr = None
		self._tCs = None
		self._rCc = None
		self._rCr = None
		self._rCs = None
		self._A   = None
		self._V   = None
		self._W   = None
		self._M   = None

	# Use nBatches & BatchSize to determine storage arr shapes ⟶ assign empty arrs to batched storage
	# (they get populated later)
	cdef void     __allocate_batched_storage( self ): #noexcept:

		cdef tuple Hshape  = ( self.nBatches, self.BatchSize, self.Lmax, EVEC_SIZE ),                                  \
				   Mshape  = ( self.nBatches, self.BatchSize, self.Lmax ),                                             \
				   hCshape = ( self.nBatches, self.BatchSize, MAX_HOLE_CARDS ),                                        \
				   fCshape = ( self.nBatches, self.BatchSize, FLOP_DEAL_SIZE ),                                        \
				   tCshape = ( self.nBatches, self.BatchSize, ),                                                       \
				   rCshape = ( self.nBatches, self.BatchSize, ),                                                       \
				   Ashape  = ( self.nBatches, self.BatchSize, EVEC_SIZE ),                                             \
				   Vshape  = ( self.nBatches, self.BatchSize, ),                                                       \
				   Wshape  = ( self.nBatches, self.BatchSize, )

		self.H_batched   = cyarr( Hshape,  INTSIZE,  'i' ) # batched game histories
		self.hCc_batched = cyarr( hCshape, UINTSIZE, 'I' ) # batched hole cardIDs
		self.hCr_batched = cyarr( hCshape, UINTSIZE, 'I' ) # batched hole rankIDs
		self.hCs_batched = cyarr( hCshape, UINTSIZE, 'I' ) # batched hole suitIDs
		self.fCc_batched = cyarr( fCshape, UINTSIZE, 'I' ) # batched flop cardIDs
		self.fCr_batched = cyarr( fCshape, UINTSIZE, 'I' ) # batched flop rankIDs
		self.fCs_batched = cyarr( fCshape, UINTSIZE, 'I' ) # batched flop suitIDs
		self.tCc_batched = cyarr( tCshape, UINTSIZE, 'I' ) # batched turn cardIDs
		self.tCr_batched = cyarr( tCshape, UINTSIZE, 'I' ) # batched turn rankIDs
		self.tCs_batched = cyarr( tCshape, UINTSIZE, 'I' ) # batched turn suitIDs
		self.rCc_batched = cyarr( rCshape, UINTSIZE, 'I' ) # batched river cardIDs
		self.rCr_batched = cyarr( rCshape, UINTSIZE, 'I' ) # batched river rankIDs
		self.rCs_batched = cyarr( rCshape, UINTSIZE, 'I' ) # batched river suitIDs
		self.A_batched   = cyarr( Ashape,  UINTSIZE, 'I' ) # batched action vectors
		self.V_batched   = cyarr( Vshape,  FLTSIZE,  'f' ) # batched adv targets
		self.W_batched   = cyarr( Wshape,  UINTSIZE, 'I' ) # batched sample weights
		self.M_batched   = cyarr( Mshape,  UINTSIZE, 'I' ) # batched history masks

	# ∀ batch: find start & end idx of batch slices ⟶ copy data subset from temp storage to batched storage
	cdef void     __partition_storage( self ): #noexcept:

		self.__allocate_batched_storage()  # initialize empty batch arrays with correct shapes
		
		cdef uint b, bIdx, bsize, bstart, bstop
		cdef bint Batch_Not_Irregular

		for b from 0 <= b < self.nBatches:
			Batch_Not_Irregular = (b != self.nBatches-1) or (self._batches_uniform)
			bsize  = self.BatchSize if Batch_Not_Irregular else self._final_bsize
			bstart = self.BatchSize * b
			bstop  = bstart + bsize

			self.H_batched  [ b ][ :bsize ] = self._H  [ bstart:bstop ] # histories
			self.hCc_batched[ b ][ :bsize ] = self._hCc[ bstart:bstop ] # hole cardIDs
			self.hCr_batched[ b ][ :bsize ] = self._hCr[ bstart:bstop ] # hole rankIDs
			self.hCs_batched[ b ][ :bsize ] = self._hCs[ bstart:bstop ] # hole suitIDs
			self.fCc_batched[ b ][ :bsize ] = self._fCc[ bstart:bstop ] # flop cardIDs
			self.fCr_batched[ b ][ :bsize ] = self._fCr[ bstart:bstop ] # flop rankIDs
			self.fCs_batched[ b ][ :bsize ] = self._fCs[ bstart:bstop ] # flop suitIDs
			self.tCc_batched[ b ][ :bsize ] = self._tCc[ bstart:bstop ] # turn cardIDs
			self.tCr_batched[ b ][ :bsize ] = self._tCr[ bstart:bstop ] # turn rankIDs
			self.tCs_batched[ b ][ :bsize ] = self._tCs[ bstart:bstop ] # turn suitIDs
			self.rCc_batched[ b ][ :bsize ] = self._rCc[ bstart:bstop ] # river cardIDs
			self.rCr_batched[ b ][ :bsize ] = self._rCr[ bstart:bstop ] # river rankIDs
			self.rCs_batched[ b ][ :bsize ] = self._rCs[ bstart:bstop ] # river suitIDs
			self.A_batched  [ b ][ :bsize ] = self._A  [ bstart:bstop ] # action vectors
			self.V_batched  [ b ][ :bsize ] = self._V  [ bstart:bstop ] # adv targets
			self.W_batched  [ b ][ :bsize ] = self._W  [ bstart:bstop ] # sample weights
			self.M_batched  [ b ][ :bsize ] = self._M  [ bstart:bstop ] # history masks

	# Just so we can check that all the metadata looks good, mostly a testing/debugging tool
	# TODO: _hC and _bC were split into rank, suit, and card idx fields, fix this
	cdef void     __constructor_summary( self ): #noexcept:

		cdef bint H_Exists  = self._H  is not None,                                                                    \
				  hC_Exists = self._hC is not None,                                                                    \
				  bC_Exists = self._bC is not None,                                                                    \
				  A_Exists  = self._A  is not None,                                                                    \
				  V_Exists  = self._V  is not None,                                                                    \
				  W_Exists  = self._W  is not None,                                                                    \
				  M_Exists  = self._M  is not None

		cdef str                                                                                                       \
			header = "\n"+(f"="*50)+"\n"+f"DATAMACHINE CONSTRUCTED @ RANK {self.RANK}".center(50),                     \
			wsStr  = f"WORLD_SIZE = {self.WORLD_SIZE}\n",                                                              \
			rStr   = f"RANK       = {self.RANK}      \n",                                                              \
			dvStr  = f"GPU        = {self.GPU}       \n",                                                              \
			stStr  = f"_dataStart = {self._dataStart}\n",                                                              \
			spStr  = f"_dataStop  = {self._dataStop} \n",                                                              \
			bsStr  = f"BatchSize  = {self.BatchSize} \n",                                                              \
			ntStr  = f"nSamples   = {self.nSamples}  \n",                                                              \
			nbStr  = f"nBatches   = {self.nBatches}  \n",                                                              \
			fbStr  = f"_final_bsize     = {self._final_bsize} \n",                                                     \
			beStr  = f"_batches_uniform = {self._batches_uniform}\n",                                                  \
			hStr   = f"\t_H:  {self._H.shape  if H_Exists  else None}, H_batched:  {self.H_batched.shape}\n",          \
			hcStr  = f"\t_hC: {self._hC.shape if hC_Exists else None}, hC_batched: {self.hC_batched.shape}\n",         \
			bcStr  = f"\t_bC: {self._bC.shape if bC_Exists else None}, bC_batched: {self.bC_batched.shape}\n",         \
			aStr   = f"\t_A:  {self._A.shape  if A_Exists  else None}, A_batched:  {self.A_batched.shape}\n",          \
			vStr   = f"\t_V:  {self._V.shape  if V_Exists  else None}, V_batched:  {self.V_batched.shape}\n",          \
			wStr   = f"\t_W:  {self._W.shape  if W_Exists  else None}, W_batched:  {self.W_batched.shape}\n",          \
			mStr   = f"\t_M:  {self._M.shape  if M_Exists  else None}, M_batched:  {self.M_batched.shape}\n",          \
			shStr  = f"STORAGE SHAPES:\n" + hStr + mStr + hcStr + hcStr +aStr + vStr + wStr,                           \
			info   = f'\n' + wsStr + rStr + dvStr + stStr + spStr + bsStr + ntStr + nbStr + fbStr + beStr + shStr,     \
			tail   = f"="*50

		print( header+info+tail )

	cdef DataBatch _get_batch( self, uint bIdx ): #noexcept:

		cdef:
			bint  Batch_Not_Irregular = (bIdx != self.nBatches-1) or (self._batches_uniform) 
			uint  bsize = self.BatchSize if Batch_Not_Irregular else self._final_bsize
			int3  bH    = self.H_batched  [ bIdx,:bsize ]
			uint2 bhCc  = self.hCc_batched[ bIdx,:bsize ],                                                             \
				  bhCr  = self.hCr_batched[ bIdx,:bsize ],                                                             \
				  bhCs  = self.hCs_batched[ bIdx,:bsize ],                                                             \
				  bfCc  = self.fCc_batched[ bIdx,:bsize ],                                                             \
				  bfCr  = self.fCr_batched[ bIdx,:bsize ],                                                             \
				  bfCs  = self.fCs_batched[ bIdx,:bsize ]
			uint1 btCc  = self.tCc_batched[ bIdx,:bsize ],                                                             \
				  btCr  = self.tCr_batched[ bIdx,:bsize ],                                                             \
				  btCs  = self.tCs_batched[ bIdx,:bsize ],                                                             \
				  brCc  = self.rCc_batched[ bIdx,:bsize ],                                                             \
				  brCr  = self.rCr_batched[ bIdx,:bsize ],                                                             \
				  brCs  = self.rCs_batched[ bIdx,:bsize ]
			uint2 bA    = self.A_batched  [ bIdx,:bsize ]
			flt1  bV    = self.V_batched  [ bIdx,:bsize ]
			uint1 bW    = self.W_batched  [ bIdx,:bsize ]
			uint2 bM    = self.M_batched  [ bIdx,:bsize ]

		# TODO: Seriously this function signature is a warcrime, roll this up into a tuple or something
		return DataBatch( bH, bhCc,bhCr,bhCs, bfCc,bfCr,bfCs, btCc,btCr,btCs, brCc,brCr,brCs, 
						  bA, bV, bW, bM, self.GPU )

	# TODO: C_batched was split into rank, suit, and card idx fields, fix this
	cdef void      _summary( self ): #noexcept:

		print( '\n'+('='*50) )
		print( "DATAMACHINE SUMMARY".center(50) )
		print( ('='*50)+'\n' )
		print( f"\tWORLD_SIZE:........{self.WORLD_SIZE}" )
		print( f"\tRANK:..............{self.RANK}" )
		print( f"\tGPU:...............{self.GPU}" )
		print( f"\tSamples:...........{self.nSamples}" )
		print( f"\tBatchSize:.........{self.BatchSize}" )
		print( f"\tnBatches:..........{self.nBatches}" )
		print( f"\t_final_bsize:......{self._final_bsize}" )
		print( f"\t_batches_uniform:..{self._batches_uniform}" )
		print( f"\thLenMax:...........{self.Lmax}" )
		print( f"\tH_batched.shape:...{tuple( self.H_batched.shape )[ :self.H_batched.ndim ]}" )
		print( f"\tC_batched.shape:...{tuple( self.C_batched.shape )[ :self.C_batched.ndim ]}" )
		print( f"\tA_batched.shape:...{tuple( self.A_batched.shape )[ :self.A_batched.ndim ]}" )
		print( f"\tV_batched.shape:...{tuple( self.V_batched.shape )[ :self.V_batched.ndim ]}" )
		print( f"\tW_batched.shape:...{tuple( self.W_batched.shape )[ :self.W_batched.ndim ]}" )
		print( f"\tM_batched.shape:...{tuple( self.M_batched.shape )[ :self.M_batched.ndim ]}" )


	# ---------- PYTHON INTERFACE ----------------------------------------------
	# These are all we need to expose to the pure-python NN training loop.
	# Everything else is a C-level internal helper to enable us to expose batches.

	def get_batch( self, uint bIdx ): 
		return self._get_batch( bIdx )

	def summary( self ): 
		self._summary()


# Just stores a bunch of metadata about CFR runs. Not just for human consumption; various flags
# etc in here are read during CFR orchetration to determine when to start different phases and
# do various file management ops. 
cdef class CFR_metadata:

	# Constructor for case of pre-existing metadata, allows us to persist across runs
	@staticmethod 
	cdef CFR_metadata load( str from_file ): #noexcept:
		with open( from_file, 'rb' ) as metafile: 
			return pickle.load( metafile )

	def __init__( self, str metafile ): 
		self.__INIT__( metafile )
		
	# The lists here hold metadata values per CFR iteration
	# TODO: Some of these vars really should be renamed
	cdef void __INIT__( self, str metafile ): #noexcept:

		self.METAFILE              = metafile
		self.CFRItersCompleted     = 0
		self.CurrentIter           = 1
		self.Iter_CPhase_Completed = 0

		# Traversal phase (aka KPhase) stuff
		self.nTravsDone       = []
		self.nNodesSeen       = []
		self.TreeComplexities = []
		self.KDurs_s          = [] # serialized KPhase durations, for speedup comparison vs parallel
		self.KDurs_p          = [] # real parallel total KPhase durations
		self.kTimeIsoAvgs_s   = [] # serialized avg time taken for one traversal
		self.kTimeIsoAvgs_p   = [] # real parallel avg time taken for one traversal

		# Calc phase (aka aPhase) stuff ("collection phase" = KPhase + aPhase)
		self.nSolvedPositions  = []
		self.nSolvedSubgames   = []
		self.nCollectedSamples = []
		self.ExplorationDepths = [] # solved positions in each game round
		self.aDurs_s           = [] # serialized adv target calc time, for comparison vs parallel
		self.aDurs_p           = [] # real parallel adv target calcualtion time
		self.ColDurs_s         = [] # serialized total collection phase duration
		self.ColDurs_p         = [] # real parallel total collection phase duration

		# Train phase (aka nnPhase) stuff
		self.TrainDurations = []
		self.InitLosses     = []
		self.EndLosses      = []
		self.MinLosses      = []
		self.MaxLosses      = []
		self.MinLossInds    = [] # epochs where min loss achieved
		self.MaxLossInds    = [] # epochs where max loss achieved

		# Overall iter stuff
		self.tDurs_s        = [] # serialized total iter durations, for speedup comparison vs parallel
		self.tDurs_p        = [] # real parallel total iter durations
		self.kTimeAggAvgs_s = [] # "aggregate" CFR iter time per trav (tDur_s/nTravsDone), serialized
		self.kTimeAggAvgs_p = [] # "aggregate" CFR iter time per trav (tDur_p/nTravsDone), true parallel

	# TODO: This involves another dangerous clear-then-save operation. Temp backup file before clear?
	cdef void   save( self ): #noexcept:
		open( self.METAFILE,'wb+' ).close()  # clear everything current metadata file, prob unnecessary but whatev
		with open( self.METAFILE,'wb' ) as metafile:
			pickle.dump( self, metafile, protocol=-1 )
		
	# CFR iters are done in worker segments - each keeps their own records so need to get those first
	cdef list __get_segment_files( self, str recordDir ): #noexcept:
		cdef str segfile
		return [ recordDir + '/' + segfile for segfile in listdir( recordDir ) ]

	# Extracts metadata dicts from segmented individual worker record files
	cdef list __get_segment_dicts( self, str recordDir ): #noexcept:

		cdef:
			dict segdict
			str  sfile
			list segfiles = self.__get_segment_files( recordDir ), segdicts=[]

		for sfile in segfiles:
			with open( sfile,'rb' ) as segfile:
				segdict = pickle.load( segfile )
			segdicts.append( segdict )

		return segdicts

	# Combines segmented worker metadata dicts into one
	cdef dict __unify_segment_dicts( self, list segdicts ): #noexcept:

		cdef uint segs = len( segdicts ), r, p
		cdef dict segdict, iterDict = {}

		iterDict[ 'nTravsDone' ]                 = sum([ segdict[ 'SegmentTravsDone' ]  for segdict in segdicts ])
		iterDict[ 'nSolvedPositions' ]           = sum([ segdict[ 'nSolvedPositions' ]  for segdict in segdicts ])
		iterDict[ 'nSolvedSubgames' ]            = sum([ segdict[ 'nSolvedSubgames' ]   for segdict in segdicts ])
		iterDict[ 'nCollectedSamples' ]          = sum([ segdict[ 'nCollectedSamples' ] for segdict in segdicts ])
		iterDict[ 'TraversalDuration_serial' ]   = sum([ segdict[ 'TraversalDuration' ] for segdict in segdicts ])
		iterDict[ 'TraversalDuration_parallel' ] = max([ segdict[ 'TraversalDuration' ] for segdict in segdicts ])
		iterDict[ 'AdvCalcDuration_serial' ]     = sum([ segdict[ 'AdvCalcTime' ]       for segdict in segdicts ])
		iterDict[ 'AdvCalcDuration_parallel' ]   = max([ segdict[ 'AdvCalcTime' ]       for segdict in segdicts ])
		iterDict[ 'CollectDuration_serial' ]     = sum([ segdict[ 'SegmentDuration' ]   for segdict in segdicts ])
		iterDict[ 'CollectDuration_parallel' ]   = max([ segdict[ 'SegmentDuration' ]   for segdict in segdicts ])
		iterDict[ 'TotalTreeComplexity' ]        = sum([ segdict[ 'TreeComplexity' ]    for segdict in segdicts ])/segs
		iterDict[ 'TravTimeIsoAvg_serial' ]      = iterDict[ 'TraversalDuration_serial' ]/iterDict[ 'nTravsDone' ]
		iterDict[ 'TravTimeIsoAvg_parallel' ]    = iterDict[ 'TraversalDuration_parallel' ]/iterDict['nTravsDone']
		iterDict[ 'ExplorationDepths' ]          = [0]*NUM_ROUNDS
		iterDict[ 'nNodesSeen' ]                 = [0]*(NUM_PLAYERS+2)

		for segdict in segdicts:

			for r from 1 <= r <= NUM_ROUNDS:
				iterDict[ 'ExplorationDepths' ][ r-1 ] += segdict[ 'ExplorationDepths' ][ r-1 ]

			for p from 0 <= p <= NUM_PLAYERS+1:
				iterDict[ 'nNodesSeen' ][ p ] += segdict[ 'nNodesSeen' ][ p ]

		return iterDict

	# Extracts and stores data from unified record dict
	cdef void __update_collection_records( self, dict iterDict ): #noexcept:
	
		self.nTravsDone.append( iterDict[ 'nTravsDone' ] )
		self.nNodesSeen.append( iterDict[ 'nNodesSeen' ] )
		self.nSolvedPositions.append( iterDict[ 'nSolvedPositions' ] )
		self.nSolvedSubgames.append( iterDict[ 'nSolvedSubgames' ] )
		self.kTimeIsoAvgs_s.append( iterDict[ 'TravTimeIsoAvg_serial' ] )
		self.kTimeIsoAvgs_p.append( iterDict[ 'TravTimeIsoAvg_parallel' ] )
		self.nCollectedSamples.append( iterDict[ 'nCollectedSamples' ] )
		self.ExplorationDepths.append( iterDict[ 'ExplorationDepths' ] )
		self.TreeComplexities.append( iterDict[ 'TotalTreeComplexity' ] )
		self.KDurs_s.append( iterDict[ 'TraversalDuration_serial' ] )
		self.KDurs_p.append( iterDict[ 'TraversalDuration_parallel' ] )
		self.aDurs_s.append( iterDict[ 'AdvCalcDuration_serial' ] )
		self.aDurs_p.append( iterDict[ 'AdvCalcDuration_parallel' ] )
		self.ColDurs_s.append( iterDict[ 'CollectDuration_serial' ] )
		self.ColDurs_p.append( iterDict[ 'CollectDuration_parallel' ] )

	# Lets us see info about the part of the game tree we explored and our performance while doing so
	cdef void   collection_summary( self ): #noexcept:

		cdef:
			uint  t = self.CurrentIter-1, K = self.nTravsDone[ t ], n
			float avgSamples = self.nCollectedSamples[ t ]/K 
			list  nNodes     = [ str( n ) for n in self.nNodesSeen[ t ] ],                                             \
				  avgNodes   = [ f"{(n/K):.3f}" for n in self.nNodesSeen[ t ] ] 

		print( '\n'+("="*100) )
		print( f"ITER {self.CurrentIter} COLLECTION PHASE COMPLETE".center(100) )
		print( "="*100 )
		print( f"\tTotal collection time (actual):     {util.HMS( <float>self.KDurs_s[ t ] )}" )
		print( f"\tTotal collection time (serialized): {util.HMS( <float>self.KDurs_p[ t ] )}" )
		print( f"\tIso avg trav time (actual):         {self.kTimeIsoAvgs_p[ t ]:.7f}sec" )
		print( f"\tIso avg trav time (serialized):     {self.kTimeIsoAvgs_s[ t ]:.7f}sec" )
		print( f"\tNum nodes encountered:              [ {' | '.join( nNodes )} ], [ {' | '.join( avgNodes )} ]/trav)" )
		print( f"\tSamples collected:                  {self.nCollectedSamples[ t ]} (avg {avgSamples:.3f}/trav)" )
		print( f"\tTotal positions solved:             {self.nSolvedPositions[ t ]}" )
		print( f"\tTotal subgames solved:              {self.nSolvedSubgames[ t ]}" )
		print( f"\tFully explored position depth spread:" )
		print( f"\t\tPreflop: {self.ExplorationDepths[ t ][ PREFLOP ]}" )
		print( f"\t\tFlop:    {self.ExplorationDepths[ t ][ FLOP ]}" )
		print( f"\t\tTurn:    {self.ExplorationDepths[ t ][ TURN ]}" )
		print( f"\t\tRiver:   {self.ExplorationDepths[ t ][ RIVER ]}" )

	# On completion of iter collect phase: load segmented mdata ⟶ unify & save it ⟶ destroy segmented data
	cdef void  _collection_phase_completed( self, str recordDir ): #noexcept:

		print( f"\nLoading segment record dicts from segrec dir: {recordDir}..." )
		cdef list segdicts = self.__get_segment_dicts( recordDir )
		print( f"{len( segdicts )} segrec dicts loaded successfully." )

		print( f"\nUnifying segmented records..." )
		cdef dict iterDict = self.__unify_segment_dicts( segdicts )
		print( f"Segmented records unified successfully" )

		self.__update_collection_records( iterDict ) # Add this iter's mdata to pre-existing mdata
		self.collection_summary()
		self.Iter_CPhase_Completed = TRUE
		self.save()

	# Records data from all phases of a CFR iter. God have mercy on my soul for this function
	# TODO: Is there a way to make this less satanic?
	cdef void   record_run( self, str dataDir ): #noexcept:

		cdef:
			uint2 nNodesSeen = NP( self.nNodesSeen,dtype=uintc )

			uint                                                                                                       \
				NODETOTALIDX  = nNodesSeen.shape[1]-1,                                                                 \
				runTotalNodes = <uint>sum( nNodesSeen[ :,NODETOTALIDX ] ),                                             \
				runTotalTime  = <uint>sum( self.tDurs_p ),                                                             \
				T = self.CFRItersCompleted, K = sum( self.nTravsDone ), t, tTime

			object now   = datetime.datetime.now()
			list   output= [], Ts = [ " `,:/ " ] 

			double                                                                                                     \
				avgIterTime   = runTotalTime/T,                                                                        \
				avgTravTime_s = sum( self.tDurs_s )/K,                                                                 \
				avgTravTime_p = sum( self.tDurs_p )/K

			int1                                                                                                       \
				tNums = np.arange( T,dtype=intc )+1,                                                                   \
				solPF = NP([ self.ExplorationDepths[ t ][ 0 ] for t in range( T ) ],dtype=intc ),                      \
				solF  = NP([ self.ExplorationDepths[ t ][ 1 ] for t in range( T ) ],dtype=intc ),                      \
				solT  = NP([ self.ExplorationDepths[ t ][ 2 ] for t in range( T ) ],dtype=intc ),                      \
				solR  = NP([ self.ExplorationDepths[ t ][ 3 ] for t in range( T ) ],dtype=intc )

			dbl1                                                                                                       \
				cumulTraversals  = np.cumsum( NP( self.nTravsDone ),dtype=f64 ),                                       \
				cumulTotalTime_s = np.cumsum( NP( self.tDurs_s ),dtype=f64 ),                                          \
				cumulTotalTime_p = np.cumsum( NP( self.tDurs_p ),dtype=f64 ),                                          \
				tDurRAvg_s       = NP( cumulTotalTime_s ) / NP( tNums ),                                               \
				tDurRAvg_p       = NP( cumulTotalTime_p ) / NP( tNums ),                                               \
				kTimeRAvg_s      = NP( cumulTotalTime_s ) / NP( cumulTraversals ),                                     \
				kTimeRAvg_p      = NP( cumulTotalTime_p ) / NP( cumulTraversals ),                                     \
				eTimes           = NP( self.TrainDurations )/TRAIN_EPOCHS

			dbl2                                                                                                       \
				lChanges = ((NP( self.InitLosses )-NP( self.EndLosses ))/NP( self.InitLosses ))*100,                   \
				lSpreads = ((NP( self.MaxLosses )-NP( self.MinLosses ))/NP( self.MaxLosses ))*100

			str                                                                                                        \
				line, p, s,                                                                                            \
				completedOn = f"{now.day}/{now.month}/{now.year}",                                                     \
				completedAt = f"{now.time()}",                                                                         \
				titleBar    =                                                                                          \
					'\n\n' + "="*100 +'\n' + f"RECORDS FOR CFR RUN OF {T} ITERS".center(100) +'\n' + "="*100 +'\n',    \
				cStr        = f"\tIteration {T} completed on {completedOn} at {completedAt} GMT",                      \
				nStr        = f"\tRun total nodes encountered: {runTotalNodes} ( avg {(runTotalNodes/T):.0f}/iter) ",  \
				tStr        = f"\tRun total time taken:                     {util.HMS( runTotalTime )}",               \
				tptStr      = f"\tRunning avg time per iter:                {util.HMS( avgIterTime )}",                \
				tpkStr_p    = f"\tActual running avg agg time per trav:     {avgTravTime_p:.5f}s",                     \
				tpkStr_s    = f"\tSerialized running avg agg time per trav: {avgTravTime_s:.5f}s"

		for t from 1 <= t <= T:
			Ts.append( '\t' + (f"Iteration {t}: ".ljust(14)) )
		output.extend([ titleBar, cStr, nStr, tStr, tptStr, tpkStr_p, tpkStr_s ])

		output.append( "\nIteration durations [ actual | serialized ]:" )
		for t from 1 <= t <= T:
			p = util.HMS( self.tDurs_p[ t-1 ] ).rjust( 8 )
			s = util.HMS( self.tDurs_s[ t-1 ] ).rjust( 8 )
			output.append( Ts[t] + f"[ {p} | {s} ]" )
		output.append( f"\tSeries data ( actual ):     {util.series( self.tDurs_p,0 )}" )
		output.append( f"\tSeries data ( serialized ): {util.series( self.tDurs_s,0 )}" )

		output.append( "\nIteration phase timing breakdown:" )
		for t from 1 <= t <= T:
			tTime = <uint>self.tDurs_p[ t-1 ]
			output.append( Ts[ t ] )
			output.append( f"\t\tTraversal phase:   {((self.KDurs_p[ t-1 ]/tTime)*100):.2f}%" )
			output.append( f"\t\tCalculation phase: {((self.aDurs_p[ t-1 ]/tTime)*100):.2f}%" )
			output.append( f"\t\tNN Training phase: {((self.TrainDurations[ t-1 ]/tTime)*100):.2f}%" )

		output.append( "\nIteration time running average [ actual | serialized ]:" )
		for t from 1 <= t <= T:
			p = util.HMS( tDurRAvg_p[ t-1 ] ).rjust( 8 )
			s = util.HMS( tDurRAvg_s[ t-1 ] ).rjust( 8 )
			output.append( Ts[t] + f"[ {p} | {s} ]" )
		output.append( f"\tSeries data for regression: {util.series( list( tDurRAvg_p ),0 )}" )

		output.append( "\nTraversal durations [ actual | serialized ]:" )
		for t from 1 <= t <= T:
			p = f"{util.HMS(self.KDurs_p[ t-1 ])}".rjust( 8 )
			s = f"{util.HMS(self.KDurs_s[ t-1 ])}".rjust( 8 )
			output.append( Ts[t] + f"[ {p} | {s} ]" )
		output.append( f"\tSeries data ( actual ):     {util.series( self.KDurs_p,0 )}" )
		output.append( f"\tSeries data ( serialized ): {util.series( self.KDurs_s,0 )}" )

		output.append( "\nIso avg time per traversal [ actual | serialized ]:" )
		for t from 1 <= t <= T: 
			p = f"{self.kTimeIsoAvgs_p[ t-1 ]:.7f}s"
			s = f"{self.kTimeIsoAvgs_s[ t-1 ]:.7f}s"
			output.append( Ts[t] + f"[ {p} | {s} ]" )
		output.append( f"\tSeries data ( actual ):     {util.series( self.kTimeIsoAvgs_p,7 )}" )
		output.append( f"\tSeries data ( serialized ): {util.series( self.kTimeIsoAvgs_s,7 )}" )

		output.append( "\nAggregate traversal time running avg [ actual | serialized ]:" )
		for t from 1 <= t <= T: 
			output.append( Ts[t] + f"[ {self.kTimeAggAvgs_p[ t-1 ]:.7f}s | {self.kTimeAggAvgs_s[ t-1 ]:.7f}s ]" )
		output.append( f"\tSeries data ( actual ):     {util.series( self.kTimeAggAvgs_p,7 )}" )
		output.append( f"\tSeries data ( serialized ): {util.series( self.kTimeAggAvgs_s,7 )}" )

		output.append( "\nNodes encountered:" )
		for t from 1 <= t <= T:
			output.append( Ts[t] + f"{self.nNodesSeen[ t-1 ]}" )
		output.append( f"\tSeries data: {util.series( list( nNodesSeen[ :,NODETOTALIDX ] ),0 )}" )

		output.append( "\nTree complexity:" )
		for t from 1 <= t <= T: 
			output.append( Ts[t] + f"{self.TreeComplexities[ t-1 ]}" )
		output.append( f"\tSeries data: {util.series( self.TreeComplexities,3 )}" )

		output.append( "\nPositions solved:" )
		for t from 1 <= t <= T: 
			output.append( Ts[t] + f"{self.nSolvedPositions[ t-1 ]}" )
		output.append( f"\tSeries data: {util.series( self.nSolvedPositions,0 )}" )

		output.append( "\nSubgames solved:" )
		for t from 1 <= t <= T: 
			output.append( Ts[t] + f"{self.nSolvedSubgames[ t-1 ]}" )
		output.append( f"\tSeries data: {util.series( self.nSolvedSubgames,0 )}" )

		output.append( "\nFully explored position depth spreads:" )
		for t from 1 <= t <= T: 
			output.append( f"{Ts[t]}" )
			output.append( f"\t\tPreflop: {solPF[ t-1 ]}" )
			output.append( f"\t\tFlop:    {solF[ t-1 ]}" )
			output.append( f"\t\tTurn:    {solT[ t-1 ]}" )
			output.append( f"\t\tRiver:   {solR[ t-1 ]}" )
		output.append( f"\tSeries data (preflop): {util.series( list( solPF ),0 )}" )
		output.append( f"\tSeries data (flop):    {util.series( list( solF ) ,0 )}" )
		output.append( f"\tSeries data (turn):    {util.series( list( solT ) ,0 )}" )
		output.append( f"\tSeries data (river):   {util.series( list( solR ) ,0 )}" )

		output.append( "\nTargets collected:" )
		for t from 1 <= t <= T: 
			output.append( Ts[t] + f"{self.nCollectedSamples[ t-1 ]}" )
		output.append( f"\tSeries data: {util.series( self.nCollectedSamples,0 )}" )

		output.append( "\nTarget calculation time [ actual | serialized ]:" )
		for t from 1 <= t <= T: 
			p = f"{self.aDurs_p[ t-1 ]:.3f}sec".rjust( 10 )
			s = f"{self.aDurs_s[ t-1 ]:.3f}sec".rjust( 10 )
			output.append( Ts[t] + f"[ {p} | {s} ]" )
		output.append( f"\tSeries data ( actual ):     {util.series( self.aDurs_p,3 )}" )
		output.append( f"\tSeries data ( serialized ): {util.series( self.aDurs_s,3 )}" )

		output.append( "\nNN training durations:" )
		for t from 1 <= t <= T: 
			output.append( Ts[t] + f"{util.HMS( self.TrainDurations[ t-1 ])}" )
		output.append( f"\tSeries data: {util.series( self.TrainDurations,0 )}" )

		output.append( "\nAvg NN training epoch time:" )
		for t from 1 <= t <= T:
			output.append( Ts[t] + f"{eTimes[ t-1 ]:.5f}s" )
		output.append( f"\tSeries data: {util.series( list( eTimes ),5 )}" )

		output.append( "\nInitial [ loss | vloss ] values:" )
		for t from 1 <= t <= T: 
			output.append( Ts[t] + f"[ {self.InitLosses[ t-1 ][0]:.7f} | {self.InitLosses[ t-1 ][1]:.7f} ]" )
		output.append( f"\tSeries data (loss):  {util.series( [ self.InitLosses[ t ][0] for t in range(T) ], 7 )}" )
		output.append( f"\tSeries data (vloss): {util.series( [ self.InitLosses[ t ][1] for t in range(T) ], 7 )}" )

		output.append( "\nFinal [ loss | vloss ] values:" )
		for t from 1 <= t <= T: 
			output.append( Ts[t] + f"[ {self.EndLosses[ t-1 ][0]:.7f} | {self.EndLosses[ t-1 ][1]:.7f} ]" )
		output.append( f"\tSeries data (loss):  {util.series( [ self.EndLosses[ t ][0] for t in range(T) ], 7 )}" )
		output.append( f"\tSeries data (vloss): {util.series( [ self.EndLosses[ t ][1] for t in range(T) ], 7 )}" )

		output.append( "\nOverall loss changes:" )
		for t from 1 <= t <= T: 
			output.append( Ts[t] + f"[ {lChanges[ t-1,0 ]:.2f}% | {lChanges[ t-1,1 ]:.2f}% ]" )
		output.append( f"\tSeries data (loss):  {util.series( list( lChanges[:,0] ),2 )}" )
		output.append( f"\tSeries data (vloss): {util.series( list( lChanges[:,1] ),2 )}" )

		output.append( "\nMax [ loss | vloss ] values:" )
		for t from 1 <= t <= T: 
			output.append( Ts[t] + f"[ {self.MaxLosses[ t-1 ][0]:.7f} | {self.MaxLosses[ t-1 ][1]:.7f} ]" )
		output.append( f"\tSeries data (loss):  {util.series( [ self.MaxLosses[ t ][0] for t in range(T) ], 7 )}" )
		output.append( f"\tSeries data (vloss): {util.series( [ self.MaxLosses[ t ][1] for t in range(T) ], 7 )}" )

		output.append( "\nMax [ loss | vloss ] epochs:" )
		for t from 1 <= t <= T: 
			output.append( Ts[t] + f"[ {self.MaxLossInds[ t-1 ][0]} | {self.MaxLossInds[ t-1 ][1]} ]" )
		output.append( f"\tSeries data (loss):  {util.series( [ self.MaxLossInds[ t ][0] for t in range(T) ], 0 )}" )
		output.append( f"\tSeries data (vloss): {util.series( [ self.MaxLossInds[ t ][1] for t in range(T) ], 0 )}" )

		output.append( "\nMin [ loss | vloss ] values:" )
		for t from 1 <= t <= T: 
			output.append( Ts[t] + f"[ {self.MinLosses[ t-1 ][0]:.7f} | {self.MinLosses[ t-1 ][1]:.7f} ]" )
		output.append( f"\tSeries data (loss):  {util.series( [ self.MinLosses[ t ][0] for t in range(T) ], 7 )}" )
		output.append( f"\tSeries data (vloss): {util.series( [ self.MinLosses[ t ][1] for t in range(T) ], 7 )}" )

		output.append( "\nMin [ loss | vloss ] epochs:" )
		for t from 1 <= t <= T: 
			output.append( Ts[t] + f"[ {self.MinLossInds[ t-1 ][0]} | {self.MinLossInds[ t-1 ][1]} ]" )
		output.append( f"\tSeries data (loss):  {util.series( [ self.MinLossInds[ t ][0] for t in range(T) ], 0 )}" )
		output.append( f"\tSeries data (vloss): {util.series( [ self.MinLossInds[ t ][1] for t in range(T) ], 0 )}" )

		output.append( "\nActual [ loss | vloss ] spreads:" )
		for t from 1 <= t <= T: 
			output.append( Ts[t] + f"[ {lSpreads[ t-1,0 ]:.2f}% | {lSpreads[ t-1,1 ]:.2f}% ]" )
		output.append( f"\tSeries data (loss):  {util.series( list( lSpreads[:,0] ),2 )}" )
		output.append( f"\tSeries data (vloss): {util.series( list( lSpreads[:,1] ),2 )}" )

		output.append( f"\nIs it real?\n" )
		
		for l in range( 1,len( output )+1 ):
			output[ l-1 ] += '\n'

		cdef str outputFile = dataDir + "/RunRecords.txt"
		with open( outputFile,"w+" ) as recordfile: 
			recordfile.writelines( output )

	# Just prints the entirety of the most recently saved mdata file
	cdef void   print_latest( self ): #noexcept:

		cdef list line_by_line
		cdef str  line

		with open( self.METAFILE ) as metafile:
			line_by_line = metafile.readlines()
			
		for line in line_by_line:
			print( line )

	# End-of-iter summary 
	cdef void   print_iter( self ): #noexcept:

		cdef:
			uint  t = self.CFRItersCompleted-1, n
			float avgTargs  = self.nCollectedSamples[ t ] / self.nTravsDone[ t ],                                      \
				  tTime_s   = <float>self.tDurs_s[ t ],                                                                \
				  tTime_p   = <float>self.tDurs_p[ t ],                                                                \
				  cTime_s   = <float>self.ColDurs_s[ t ],                                                              \
				  cTime_p   = <float>self.ColDurs_p[ t ],                                                              \
				  nnTime    = <float>self.TrainDurations[ t ],                                                         \
				  nAvg
			list  nNodes    = [ str( n ) for n in self.nNodesSeen[ t ] ],                                              \
				  avgNodes  = [ f"{(n/self.nTravsDone[ t ]):.3f}" for n in self.nNodesSeen[ t ] ]

		print( '\n'+("="*100) )
		print( f"CFR ITERATION {self.CFRItersCompleted} COMPLETE".center(100) )
		print( ("="*100) )
		
		print( f"Total time taken (actual):      {util.HMS( tTime_p )}" )
		print( f"Total time taken (serialized):  {util.HMS( tTime_s )}" )
		
		print( f"Iso avg trav time (actual):     {self.kTimeIsoAvgs_p[ t ]:.5f}sec" )
		print( f"Iso avg trav time (serialized): {self.kTimeIsoAvgs_s[ t ]:.5f}sec" )
		print( f"Agg avg trav time (actual):     {self.kTimeAggAvgs_p[ t ]:.5f}sec" )
		print( f"Agg avg trav time (serialized): {self.kTimeAggAvgs_s[ t ]:.5f}sec" )

		print( f"Collection time (actual):       {util.HMS( cTime_p )} ( {((cTime_p/tTime_p)*100):.2f}% of total )" )
		print( f"Collection time (serialized):   {util.HMS( cTime_s )}" )

		print( f"NN training time:               {util.HMS( nnTime )} ( {((nnTime/tTime_p)*100):.2f}% of total )" )

		print( f"Unique nodes seen:              [ { ' | '.join( nNodes ) } ], [ {' | '.join( avgNodes )} ]/trav ")
		print( f"Total tree complexity:          {self.TreeComplexities[ t ]:.3f}" )
		print( f"Targets collected:              {self.nCollectedSamples[ t ]} (avg {avgTargs:.3f}/trav)" )
		print( f"Total positions solved:         {self.nSolvedPositions[ t ]}" )
		print( f"Total subgames solved:          {self.nSolvedSubgames[ t ]}" )
		
		print( f"Depth spread of fully explored positions:" )
		print( f"\tPreflop: {self.ExplorationDepths[ t ][ PREFLOP ]}" )
		print( f"\tFlop:    {self.ExplorationDepths[ t ][ FLOP ]}" )
		print( f"\tTurn:    {self.ExplorationDepths[ t ][ TURN ]}" )
		print( f"\tRiver:   {self.ExplorationDepths[ t ][ RIVER ]}" )

	# Singular call at the end of a CFR iteration to handle all end-of-iter record-keeping
	cdef void  _CFR_iteration_completed( self, double trainTime, list lHist, list vlHist, str dataDir ): #noexcept:

		# Calculate derived quantities
		cdef:
			uint                                                                                                       \
				t = self.CFRItersCompleted,                                                                            \
				E = <uint>len( lHist ),                                                                                \
				PARALLEL_TRAIN_SPEEDUP_FACTOR=5 # empirical approx speedup factor on 8x GPU vs 1x
			float                                                                                                      \
				trainTime_s = trainTime * PARALLEL_TRAIN_SPEEDUP_FACTOR,                                               \
				iterTime_s  = self.ColDurs_s[ t ] + trainTime_s,                                                       \
				iterTime_p  = self.ColDurs_p[ t ] + trainTime,                                                         \
				travAvg_s   = iterTime_s / self.nTravsDone[ t ],                                                       \
				travAvg_p   = iterTime_p / self.nTravsDone[ t ],                                                       \
				initLoss    = lHist[0]  if lHist[0]  > 0 else 1e-10,                                                   \
				initVLoss   = vlHist[0] if vlHist[0] > 0 else 1e-10,                                                   \
				maxLoss     = max( lHist )  if max( lHist )  > 0 else 1e-10,                                           \
				maxVLoss    = max( vlHist ) if max( vlHist ) > 0 else 1e-10,                                           \
				endLoss     = lHist[ E-1 ], endVLoss = vlHist[ E-1 ],                                                  \
				minLoss     = min( lHist ), minVLoss = min( vlHist )

			tuple                                                                                                      \
				initialLosses = (initLoss, initVLoss), finalLosses = (endLoss, endVLoss),                              \
				maxLosses     = (maxLoss,  maxVLoss),  minLosses   = (minLoss, minVLoss),                              \
				maxLossInds   = (lHist.index( max( lHist )), vlHist.index( max( vlHist ))),                            \
				minLossInds   = (lHist.index( min( lHist )), vlHist.index( min( vlHist )))

		# Store calculated quantities
		self.TrainDurations.append( trainTime )
		self.InitLosses.append( initialLosses )
		self.EndLosses.append( finalLosses )
		self.MinLosses.append( minLosses )     
		self.MaxLosses.append( maxLosses )
		self.MinLossInds.append( minLossInds ) 
		self.MaxLossInds.append( maxLossInds )
		self.tDurs_s.append( iterTime_s )      
		self.tDurs_p.append( iterTime_p )
		self.kTimeAggAvgs_s.append( travAvg_s )
		self.kTimeAggAvgs_p.append( travAvg_p )

		# Increment/reset per-iter counters & flags
		self.CFRItersCompleted+=1
		self.CurrentIter+=1
		self.Iter_CPhase_Completed=0

		# Save, print, and do cleanup
		self.save()
		self.record_run( dataDir )
		self.print_iter()
		print()


	# ---------- PYTHON INTERFACE FUNCTIONS --------------------------------------------------------
	# These allow us to interact with metadata from the training loop (helpful since it's the 
	# final phase of every iteration) and from simple py scripts (helpful for easily examining 
	# run data and progress).


	@staticmethod
	def pyload( str from_file ):
		return CFR_metadata.load( from_file )

	def pysave( self ):
		self.save()

	def get_solved_iters( self ):
		return self.CFRItersCompleted

	def get_current_iter( self ):
		return self.CurrentIter

	def get_current_pov( self ):
		return (( self.CurrentIter + INITIAL_POV ) % NUM_PLAYERS ) + 1

	def collection_phase_completed( self, str recordDir ):
		self._collection_phase_completed( recordDir )

	def CFR_iteration_completed( self, double trainTime, list lHist, list vlHist, str dataDir ): 
		self._CFR_iteration_completed( trainTime, lHist, vlHist, dataDir )


# *-* # 