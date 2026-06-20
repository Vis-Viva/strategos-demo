#distutils: language = c
#cython: language_level 3


cimport cython
cimport numpy as cnp
cnp.import_array()

from strategos_tools.core.CONSTS       cimport *
from strategos_tools.core.containers   cimport *
from strategos_tools.env.event_ops     cimport gameevent #DEBUG
from strategos_tools.env.gamenode_ops  cimport gamenode
from strategos_tools.env.infoset_ops   cimport infoset
from strategos_tools.env.actionset_ops cimport actionset


cdef class AdvNetInputs:

	cdef: # tensors for histories, cards, and actions
		public object H, hC_c, hC_r, hC_s, fC_c, fC_r, fC_s, tC_c, tC_r, tC_s, rC_c, rC_r, rC_s, A
		public uint   nA
		public str    GPU

	cdef void __INIT__( self, infoset I, uint GPUrank=* ) #noexcept

	cdef void __init_history( self, infoset I ) #noexcept

	cdef void __init_cards( self, infoset I ) #noexcept

	cdef void __init_actions( self, infoset I ) #noexcept

	@staticmethod
	cdef AdvNetInputs _DummyInputs( uint GPUrank=* ) #noexcept


cdef class MMInputs:

	cdef: # tensors for histories, cards, and actions
		public object H, hC_c, hC_r, hC_s, fC_c, fC_r, fC_s, tC_c, tC_r, tC_s, rC_c, rC_r, rC_s, A
		public uint   nSamples, T, nI, nA
		public str    GPU

	cdef void __INIT__( self, uint actingPlayer, infoset Ipov, uint iterSpan, uint GPUrank=* ) #noexcept

	cdef void __init_history( self, infoset Ipov, bint Opp_State ) #noexcept

	cdef void __init_cards( self, infoset Ipov, bint Opp_State ) #noexcept

	cdef void __init_actions( self, infoset Iap ) #noexcept

	@staticmethod
	cdef MMInputs _DummyInputs( uint iterSpan, uint GPUrank=* ) #noexcept
	

cdef class MMInputs_old:
	
	cdef public object H, hC, bC, A
	cdef public uint   OLD_CVEC_SIZE, OLD_EVEC_SIZE, NEW_EVEC_SIZE, HOLEVEC_SIZE, BOARDVEC_SIZE, nSamples, T, nI, nA

	cdef void  __INIT__( self, uint actingPlayer, infoset Ipov, uint iterSpan, uint GPUrank=* ) #noexcept

	cdef void  __init_tensors( self, uint3 H, uint2 hC, uint2 bC, uint2 A, uint GPUrank ) #noexcept

	cdef void  __init_pov_state( self, infoset Ipov, uint GPUrank ) #noexcept

	cdef void  __init_opp_state( self, infoset Ipov, uint GPUrank ) #noexcept

	cdef uint2 __CardConverter( self, uint[:] cardIDs ) #noexcept

	cdef uint3 __HistoryConverter( self, uint3 hArr, uint1 dealSteps ) #noexcept

	cdef uint1 __MultiCardVec( self, uint2 cardVecs ) #noexcept

	cdef uint2 __ActionsetConverter( self, uint2 A ) #noexcept


cdef class advsample:

	cdef:
		public uint    t
		public uint1   aVec
		public float   aAdv
		public infoset Infoset

	cdef void    __INIT__( self, uint iterNum, infoset I, uint1 actionVec, float actionAdv ) #noexcept

	cdef dict      to_dict( self ) #noexcept

	cdef void      save( self, str to_file ) #noexcept

	@staticmethod
	cdef advsample from_dict( dict sampleDict ) #noexcept


cdef class advmap:

	cdef:
		int     IterNum, NumSamples
		infoset Infoset
		list    ActionInds, AdvTargets

	cdef void __INIT__( self, int for_iter, infoset I, int1 aInds, flt1 aTargets ) #noexcept

	cdef list __extract_sample_dicts( self ) #noexcept

	cdef void   summary( self ) #noexcept

	cdef dict   to_dict( self ) #noexcept

	cdef void   save( self, str to_file ) #noexcept

	cdef void   save_sample_dicts( self, str to_file ) #noexcept

	@staticmethod
	cdef advmap from_dict( dict advdict ) #noexcept


cdef class DataBatch:
	
	cdef public uint   size
	cdef public object H, hCc,hCr,hCs, fCc,fCr,fCs, tCc,tCr,tCs, rCc,rCr,rCs, A, V, W, M

	cdef void __INIT__( self, int3 H,
							  uint2 hCc, uint2 hCr, uint2 hCs,
		                      uint2 fCc, uint2 fCr, uint2 fCs,
	                          uint1 tCc, uint1 tCr, uint1 tCs,
	                          uint1 rCc, uint1 rCr, uint1 rCs,
	                          uint2 A,   flt1 V,    uint1 W,   uint2 M, str GPU ) #noexcept


cdef class DATAMACHINE:

	cdef:
		public uint BatchSize, nSamples, nBatches, Lmax

		uint WORLD_SIZE, RANK, _dataStart, _dataStop, _final_bsize
		bint _batches_uniform
		str  GPU

		# temp pre-batched storage
		int3  _H
		uint2 _hCc,_hCr,_hCs, _fCc,_fCr,_fCs
		uint1 _tCc,_tCr,_tCs, _rCc,_rCr,_rCs
		uint2 _A
		flt1  _V 
		uint1 _W
		uint2 _M

		# persistent final batched storage
		int4  H_batched
		uint3 hCc_batched,hCr_batched,hCs_batched, fCc_batched,fCr_batched,fCs_batched
		uint2 tCc_batched,tCr_batched,tCs_batched, rCc_batched,rCr_batched,rCs_batched
		uint3 A_batched
		flt2  V_batched 
		uint2 W_batched
		uint3 M_batched

	cdef void     __find_max_hLen( self, list from_samples ) #noexcept

	cdef void     __determine_storage_layout( self, list shuffledSamples, uint bsize ) #noexcept

	cdef void     __allocate_temp_storage( self ) #noexcept

	cdef uint1    __history_mask( self, uint hLen ) #noexcept

	cdef void     __populate_temp_storage( self, list shuffledSamples ) #noexcept

	cdef void     __destroy_temp_storage( self ) #noexcept

	cdef void     __allocate_batched_storage( self ) #noexcept

	cdef void     __partition_storage( self ) #noexcept

	cdef void     __constructor_summary( self ) #noexcept

	cdef void     __INIT__( self, list shuffledSamples, uint bsize, int world_size, int rank ) #noexcept

	cdef void      _summary( self ) #noexcept

	cdef DataBatch _get_batch( self, uint bIdx ) #noexcept


cdef class CFR_metadata:

	cdef:
		public str  METAFILE
		public int  CurrentIter, CFRItersCompleted, IterTravsDone
		public bint Iter_CPhase_Completed
		public list nTravsDone, nNodesSeen, nSolvedPositions, nSolvedSubgames, nCollectedSamples, ExplorationDepths,   \
					kTimeIsoAvgs_s, kTimeIsoAvgs_p, KDurs_s, KDurs_p, aDurs_s, aDurs_p, ColDurs_s, ColDurs_p,          \
					InitLosses, EndLosses, MinLosses, MaxLosses, MinLossInds, MaxLossInds, TrainDurations,             \
					tDurs_s, tDurs_p, kTimeAggAvgs_s, kTimeAggAvgs_p,                                                  \
					TreeComplexities

	@staticmethod
	cdef CFR_metadata load( str from_file ) #noexcept

	cdef void __INIT__( self, str metafile ) #noexcept

	cdef void   save( self ) #noexcept

	cdef list __get_segment_files( self, str recordDir ) #noexcept

	cdef list __get_segment_dicts( self, str recordDir ) #noexcept

	cdef dict __unify_segment_dicts( self, list segdicts ) #noexcept

	cdef void __update_collection_records( self, dict iterDict ) #noexcept	
	
	cdef void   collection_summary( self ) #noexcept

	cdef void  _collection_phase_completed( self, str recordDir ) #noexcept

	cdef void   record_run( self, str dataDir ) #noexcept

	cdef void   print_latest( self ) #noexcept

	cdef void   print_iter( self ) #noexcept

	cdef void  _CFR_iteration_completed( self, double trainTime, list lHist, list vlHist, str dataDir ) #noexcept


# *-* #