#distutils: language = c
#cython: language_level 3


cimport cython
cimport numpy as cnp
cnp.import_array()

from strategos_tools.core.CONSTS        cimport *
from strategos_tools.env.gamenode_ops   cimport gamenode
from strategos_tools.env.infoset_ops    cimport infoset
from strategos_tools.env.actionset_ops  cimport actionset
from strategos_tools.utils.data_structs cimport advmap, advsample, CFR_metadata


cdef list   load_models( str from_file, int for_iter=* ) #noexcept

cdef list  _load_advmaps( str from_file, int for_iter=* ) #noexcept

cdef list __extract_all_samples( list from_advmaps ) #noexcept

cdef list __load_segmented_sample_dicts( str from_file ) #noexcept

cdef list __load_existing_sample_dicts( str from_file ) #noexcept

cdef void __clear_existing_samples( str in_file ) #noexcept

cdef void __save_unsegmented_samples( list shuffledSamples, str to_file, bint Clear_Existing=* ) #noexcept

cdef void __append_new_data( str trainFile, str valFile, list iterSamples, float val_split=* ) #noexcept

cdef void  _unsegment_iter_data( str advDir, str trainFile, str valFile, float val_split=* ) #noexcept

cdef list  _load_true_samples( str from_file ) #noexcept

cdef void  _post_iter_cleanup( str advDir, uint for_iter ) #noexcept


# *-* #