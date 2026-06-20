#distutils: language = c
#cython: language_level 3

from .card cimport Card as C
from strategos_tools.core.CONSTS cimport *

cdef dict MAX_TO_RANK_CLASS, RANK_CLASS_TO_STRING

cdef class LookupTable:
		
	cdef dict flush_lookup, unsuited_lookup

	cdef void flushes( self ) #noexcept

	cdef void straight_and_highcards( self, tuple straights, list highcards ) #noexcept

	cdef void multiples( self ) #noexcept

	cdef void write_table_to_disk( self, LookupTable table, str filepath ) #noexcept
