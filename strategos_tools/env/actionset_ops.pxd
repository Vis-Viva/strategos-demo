#distutils: language = c
#cython: language_level 3


cimport cython

from strategos_tools.core.CONSTS      cimport *
from strategos_tools.env.event_ops    cimport gameevent
from strategos_tools.env.gamenode_ops cimport gamenode
from strategos_tools.env.infoset_ops  cimport infoset

cdef class actionset:

	cdef:
		uint    Player, Posting_Blind, CurrentStack, MatchCall, MinRaise, MaxRaise, NumNonRaises, NumRaises, size,     \
				CheckCallIdx
		bint    Can_Fold, Can_Check, Can_Call, Can_Raise
		uint2   NonRaises
		infoset _I

	cdef void      __find_available_nonraises( self ) #noexcept

	cdef void      __find_available_raises( self ) #noexcept

	cdef void      __INIT__( self, infoset I ) #noexcept

	cdef gameevent   at( self, uint aIdx ) #noexcept

	cdef uint      __get_raise_aIdx( self, uint from_total_bet ) #noexcept

	cdef uint        index_of( self, gameevent e ) #noexcept

	cdef uint2       AMat( self ) #noexcept

	cdef gamenode    SourceNode( self ) #noexcept

	cdef str         inline_summary( self ) #noexcept

	cdef void        summary( self ) #noexcept

	cdef void        DIAGNOSTIC( self ) #noexcept


# *-* # 