#distutils: language = c
#cython: language_level 3
#cython: profile = False

from cpython.array cimport array as cparr
from cpython.array cimport clone as pyarr
from cython.view   cimport array as cyarr
from .lookup       cimport *

from numpy import asarray as np_arr, all as alleq


# ==================================================================================================
# Cythonized version of Deuce's hand evaluator, which uses a variant of Cactus Kev's algorithm 
# (http://www.suffecool.net/poker/evaluator.html). Extremely fast and lightweight, uses bit
# arithmetic and table lookup for all calculations.
# ==================================================================================================


cdef class Evaluator:

	def __init__( self ):
		self.lookup = LT()

	# N choose K
	cdef inline int _NcK( self, int n, int k ): #noexcept:
		return 1 if (n==k or k==0) else self._NcK( n-1,k-1 ) + self._NcK( n-1,k )	

	# Gets all possible five-card hands from supplied array of cards
	cdef int2 _get_fives( self, int1 from_cards ): #noexcept:

		cdef:
			int  nCards = from_cards.shape[ 0 ], nHands = self._NcK( nCards,MAX_HAND_SIZE ), h=0, i1, i2, i3, i4, i5
			int1 hand   = pyarr( ARR_TMPLT_i, MAX_HAND_SIZE, zero=False )
			int2 combos = cyarr( (nHands,MAX_HAND_SIZE), INTSIZE, 'i' )

		if   nCards < 5: 
			return combos[ :0 ] # :0 indeed

		elif nHands == 1: 
			combos[ 0 ] = from_cards
			return combos

		for i1 from 0 <= i1 < nCards-4:
			hand[ 0 ] = from_cards[ i1 ]

			for i2 from i1 < i2 < nCards-3:
				hand[ 1 ] = from_cards[ i2 ]

				for i3 from i2 < i3 < nCards-2:
					hand[ 2 ] = from_cards[ i3 ]

					for i4 from i3 < i4 < nCards-1:
						hand[ 3 ] = from_cards[ i4 ]

						for i5 from i4 < i5 < nCards:
							hand[ 4 ]   = from_cards[ i5 ]
							combos[ h ] = hand
							h+=1

		return combos

	# Calculates score for a five-card hand. ASSUMES INPUT CONTAINS EXACTLY 5 CARDS
	cdef inline int _get_score( self, int1 cards ): #noexcept:

		cdef int handOR

		if cards[0] & cards[1] & cards[2] & cards[3] & cards[4] & 0xF000: # Short-circuit for special case of flushes
			handOR = (cards[0] | cards[1] | cards[2] | cards[3] | cards[4]) >> 16
			return self.lookup.flush_lookup[ C.prime_product_from_rankbits( handOR ) ]

		else: # All other cases
			return self.lookup.unsuited_lookup[ C.prime_product_from_hand( cards ) ] 

	# Scores all possible five-card hands from input & returns best
	cdef inline int _get_best_score( self, int1 cards ): #noexcept:

		cdef int2 hands  = self._get_fives( cards )
		cdef int  nHands = hands.shape[ 0 ], bestScore = <int>MAX_HIGH_CARD, h, hScore

		for h from 0 <= h < nHands:
			hScore = self._get_score( hands[ h ] )
			if hScore <= bestScore: 
				bestScore = hScore

		return bestScore

	# This is what you call externally to get hand scores from an array of Deuce ints
	cdef inline uint evaluate( self, int1 cards ): #noexcept:

		cdef int nCards = cards.shape[ 0 ]

		if nCards == 5: 
			return <uint>self._get_score( cards )

		elif nCards > 5:  
			return <uint>self._get_best_score( cards )

		else:             
			return 0

	# Just gets us the best-scoring hand from the input set
	cdef int1 get_best_hand( self, int1 from_cards ): #noexcept:

		cdef:
			int  nCards = from_cards.shape[ 0 ]
			int2 hands  = self._get_fives( from_cards )
			int  nHands = hands.shape[ 0 ], bestScore = <int>MAX_HIGH_CARD, h, hScore
			int1 hand, bestHand 

		for h from 0 <= h < nHands:
			hand   = hands[ h ]
			hScore = self._get_score( hand ) 
			if hScore <= bestScore: 
				bestScore = hScore
				bestHand = hand

		return bestHand

	# Returns an identifier for the class of a hand with score hScore
	cdef int get_rank_class( self, int hScore ): #noexcept:
		if   hScore <= <int>MAX_STRAIGHT_FLUSH:  return <int>MAX_TO_RANK_CLASS[ MAX_STRAIGHT_FLUSH ]
		elif hScore <= <int>MAX_FOUR_OF_A_KIND:  return <int>MAX_TO_RANK_CLASS[ MAX_FOUR_OF_A_KIND ]
		elif hScore <= <int>MAX_FULL_HOUSE:      return <int>MAX_TO_RANK_CLASS[ MAX_FULL_HOUSE ]
		elif hScore <= <int>MAX_FLUSH:           return <int>MAX_TO_RANK_CLASS[ MAX_FLUSH ]
		elif hScore <= <int>MAX_STRAIGHT:        return <int>MAX_TO_RANK_CLASS[ MAX_STRAIGHT ]
		elif hScore <= <int>MAX_THREE_OF_A_KIND: return <int>MAX_TO_RANK_CLASS[ MAX_THREE_OF_A_KIND ]
		elif hScore <= <int>MAX_TWO_PAIR:        return <int>MAX_TO_RANK_CLASS[ MAX_TWO_PAIR ]
		elif hScore <= <int>MAX_PAIR:            return <int>MAX_TO_RANK_CLASS[ MAX_PAIR ]
		elif hScore <= <int>MAX_HIGH_CARD:       return <int>MAX_TO_RANK_CLASS[ MAX_HIGH_CARD ]

	# Just gives a human-readable hand class string corresponding to the integer class ID
	cdef str class_to_string( self, int classInt ): #noexcept:
		return RANK_CLASS_TO_STRING[ classInt ]

	# Just scales hand rank score to the range [0.0, 1.0]
	cdef float get_five_card_rank_percentage( self, int hand_rank ): #noexcept:
		return (<float>hand_rank) / (<float>MAX_HIGH_CARD)

# *-* #