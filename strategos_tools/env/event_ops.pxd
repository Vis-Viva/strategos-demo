#distutils: language = c
#cython: language_level 3


from strategos_tools.core.CONSTS cimport *


cdef class gameevent:

	cdef uint  Type, PlayedBy, RaiseAmt, BetTotal, Is_AllIn, AllInDiff, DealTo, CDealt

	cdef void  __AUTOINIT__( self, uint1 from_array ) #noexcept

	cdef void  __MANUAL_INIT__( self, uint eventType, uint playedBy, uint raiseAmt, uint betTotal, 
								uint Is_AllIn, uint allInDiff, uint dealTo, uint cDealt ) #noexcept

	cdef uint1   to_array( self ) #noexcept

	cdef str     GTString( self ) #noexcept

	cdef str     ShortString( self ) #noexcept

	cdef str     ShorterString( self, uint stepNum, bint Include_Array=* ) #noexcept

	cdef bint  __EQ__( self, gameevent e ) #noexcept

	
cdef uint2 AllNonRaises( uint player, uint callAmt=*, bint AllIn_Call=*, uint callDiff=* ) #noexcept

cdef bint  Is_Dealer_Action( uint1 e ) #noexcept

cdef bint  Is_Null( uint1 e ) #noexcept


# *-* #