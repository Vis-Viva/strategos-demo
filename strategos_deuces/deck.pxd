#distutils: language = c
#cython: language_level 3

cdef list _FULL_DECK

cdef class Deck:

	cdef list cards

	@staticmethod
	cdef list GetFullDeck() #noexcept

	cdef void shuffle( self ) #noexcept

	cdef list draw( self,int n=* ) #noexcept
