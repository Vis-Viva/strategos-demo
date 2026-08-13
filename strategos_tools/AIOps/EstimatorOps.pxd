#distutils: language = c
#cython: language_level 3

cimport cython
cimport numpy as cnp
cnp.import_array()

from strategos_tools.core.CONSTS        cimport *
from strategos_tools.env.infoset_ops    cimport infoset
from strategos_tools.env.actionset_ops  cimport actionset
from strategos_tools.utils.data_structs cimport advmap, AdvNetInputs, MMInputs, MMInputs_old, CFR_metadata

cdef str    ESTIMATOR_DEVICE
cdef object ADVNET, MULTIMODEL, ALT_MULTIMODEL


# ==================================================================================================
# GLOBAL MODEL INSTANTIATORS
# ==================================================================================================


cdef void set_estimator_device( int deviceID ) #noexcept
cdef void setup_advnet( str modelFile, uint modelIter, uint modelSize=*, bint Compiled=* ) #noexcept
cdef void setup_multimodel( str modelFile, uint iterSpan, uint modelSize=*, bint Compiled=* ) #noexcept
cdef void setup_alt_multimodel( str modelFile, uint iterSpan, uint modelSize=*, bint Compiled=* ) #noexcept


# ==================================================================================================
# SINGLE-ITER ADVNET OPS
# ==================================================================================================


cdef flt1 __AdvEstimator( infoset I ) #noexcept

cdef flt1 __ActionProbs( flt1 advI ) #noexcept

cdef flt1   Strategy( infoset I ) #noexcept


# ==================================================================================================
# MANY-ITER MULTIMODEL OPS
# ==================================================================================================


cdef flt3  __MultiAdvArray( list rawOutputs, uint T, uint nI, uint nA ) #noexcept

cdef list  __tolist( object MMoutputs ) #noexcept

cdef flt3  __LegacyMultiAdvEstimator( uint actingPlayer, infoset I, bint Alt_Model=* ) #noexcept

cdef flt3  __MultiAdvEstimator( uint actingPlayer, infoset I, bint Alt_Model=* ) #noexcept

cdef flt1  __MultiAdvSums( flt2 posMultiAdvs ) #noexcept

cdef flt2  __MultiActionProbs( flt2 multiAdvs ) #noexcept 

cdef flt3    MultiStrats( uint actingPlayer, infoset I, bint Alt_Model=*, bint Legacy_Model=* ) #noexcept

cdef dbl2    AvgStrategy( infoset I, dbl1 iterReaches ) #noexcept


