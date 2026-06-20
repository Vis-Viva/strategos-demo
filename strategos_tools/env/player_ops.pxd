#distutils: language = c
#cython: language_level 3


from strategos_tools.core.CONSTS cimport *


cdef uint1 OpponentsOf( uint player ) #noexcept

cdef bint  Are_Opponents( uint p1, uint p2 ) #noexcept

cdef uint  NextNonDealer( uint p ) #noexcept

cdef uint  PrevNonDealer( uint p ) #noexcept


# *-* #