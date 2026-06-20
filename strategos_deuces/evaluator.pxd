#distutils: language = c
#cython: language_level 3

from .card cimport Card as C
from .deck cimport Deck as D
from .lookup cimport LookupTable as LT
from strategos_tools.core.CONSTS cimport *

cdef class Evaluator:

	cdef LT lookup

	cdef int   _NcK( self, int n, int k ) #noexcept

	cdef int2  _get_fives( self, int1 from_cards ) #noexcept

	cdef int   _get_score( self, int1 cards ) #noexcept

	cdef int   _get_best_score( self, int1 cards ) #noexcept

	cdef uint   evaluate( self, int1 cards ) #noexcept

	cdef int1   get_best_hand( self, int1 from_cards ) #noexcept

	cdef int    get_rank_class( self, int hScore ) #noexcept

	cdef str    class_to_string( self, int classInt ) #noexcept

	cdef float  get_five_card_rank_percentage( self, int hand_rank ) #noexcept

