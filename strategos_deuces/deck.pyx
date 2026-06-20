#distutils: language = c
#cython: language_level 3
#cython: profile = False

from .card cimport Card as C
from random import shuffle

cdef list _FULL_DECK = []


# ==================================================================================================
# Cythonized version of Deuce's deck class. Unused by strategos, as we use our own deck entity.
# The first time the static deck is created, it's seeded with the list of unique card ints. Each 
# instantiated deck object simply copies this initial static object and shuffles it.
# ==================================================================================================


cdef class Deck:

	def __init__( self ):
		self.shuffle()

	cdef list draw( self,int n=1 ): #noexcept:
		if n == 1:
			return self.cards.pop( 0 )

		cdef list cards = []
		cdef int i
		for i from 0 <= i < n:
			cards.append( self.draw() )

		return cards

	def __str__( self ):
		return C.print_pretty_cards( self.cards )

	cdef void shuffle( self ): #noexcept:
		self.cards = Deck.GetFullDeck(); shuffle( self.cards )

	@staticmethod
	cdef list GetFullDeck(): #noexcept:

		# If full deck has already been initialized
		if len( _FULL_DECK ) > 0: 
			return _FULL_DECK 
		
		# Otherwise, initialize it
		cdef str rank, suit
		cdef int val
		for rank in C.STR_RANKS: 
			for suit,val in C.CHAR_SUIT_TO_INT_SUIT.items(): 
				_FULL_DECK.append( C.new_card( rank+suit ))

		return _FULL_DECK
