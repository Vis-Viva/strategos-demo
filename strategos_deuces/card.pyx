#distutils: language = c
#cython: language_level 3
#cython: profile = False

from termcolor import colored


# ==================================================================================================
# Cythonized version of Deuce's Card class. We interface with this so we can use Deuce's hand 
# evaluator, and for no other reason. Deuce cards are represented as 32-bit ints, so there is no 
# object instantiation - they're literally just ints. Most bits are used & have a specific meaning:
# 
# 		 		 Card:                   | 
#                                        | - p = rank prime  (two=2, three=3, four=5, ... , ace=41)
# 	    bitrank      suit&rank  prime    | - r = card rankID (two=0, three=1, four=2, ... , ace=12)
# +--------+--------+--------+--------+  | - b = bit turned on depending on rank
# |xxxbbbbb|bbbbbbbb|cdhsrrrr|xxpppppp|  | - x = bit unused
# +--------+--------+--------+--------+  | - cdhs = suit of card (bit turned on according to suit) 
#
# This representation is extremely fast and lightweight, and allows us to:
# - Make a unique prime product for each hand
# - Detect flushes
# - Detect straights
# Lots of this is inspired by http://www.suffecool.net/poker/evaluator.html
# ==================================================================================================


# Useful module consts
cdef:
	str   STR_RANKS             = '23456789TJQKA',                                                                     \
		  INT_SUIT_TO_CHAR_SUIT = 'xshxdxxxc'
	tuple INT_RANKS             = tuple( range( 13 ) ),                                                                \
		  PRIMES                = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
	dict  CHAR_RANK_TO_INT_RANK = dict( zip( tuple( STR_RANKS ), INT_RANKS ) ),                                        \
		  CHAR_SUIT_TO_INT_SUIT = { 's':1, 'h':2, 'd':4, 'c':8 },                                                      \
		  PRETTY_SUITS          = { 1 : "\u2660".encode('utf-8'),  # spades
		  							2 : "\u2665".encode('utf-8'),  # hearts
		  							4 : "\u2666".encode('utf-8'),  # diamonds
		  							8 : "\u2663".encode('utf-8') } # clubs 

cdef class Card:

	@staticmethod
	cdef inline int  new_card( str card_str ): #noexcept:

		cdef:
			str rank_char  = card_str[0], suit_char = card_str[1]
			int rank_int   = CHAR_RANK_TO_INT_RANK[ rank_char ],                                                       \
				suit_int   = CHAR_SUIT_TO_INT_SUIT[ suit_char ],                                                       \
				rank_prime = PRIMES[ rank_int ],                                                                       \
				bitrank    = 1 << rank_int << 16, suit = suit_int << 12, rank = rank_int << 8

		return bitrank | suit | rank | rank_prime

	@staticmethod
	cdef inline str  int_to_str( int card_int ): #noexcept:
		cdef int rank_int = Card.get_rank_int( card_int ), suit_int = Card.get_suit_int( card_int )
		return STR_RANKS[ rank_int ] + INT_SUIT_TO_CHAR_SUIT[ suit_int ]

	@staticmethod
	cdef inline int  get_rank_int( int card_int ): #noexcept:
		return (card_int >> 8) & 0xF

	@staticmethod
	cdef inline int  get_suit_int( int card_int ): #noexcept:
		return (card_int >> 12) & 0xF

	@staticmethod
	cdef inline int  get_bitrank_int( int card_int ): #noexcept:
		return (card_int >> 16) & 0x1FFF

	@staticmethod
	cdef inline int  get_prime( int card_int ): #noexcept:
		return card_int & 0x3F

	@staticmethod
	cdef inline list hand_to_binary( list card_strs ): #noexcept:

		cdef int  c, nCards = <int>len( card_strs )
		cdef list bhand = []
		for c from 0 <= c < nCards:
			bhand.append( Card.new_card( card_strs[ c ]) )
		return bhand

	@staticmethod
	cdef inline ll   prime_product_from_hand( int[::1] card_ints ): #noexcept:

		cdef int c, nCards = <int>card_ints.shape[0]
		cdef ll  product = 1
		for c from 0 <= c < nCards: 
			product *= (card_ints[ c ] & 0xFF)
		return product

	# Primarily used for evaluating flushes & straights, since we know all ranks there are distinct.
	@staticmethod
	cdef inline ll   prime_product_from_rankbits( int rankbits ): #noexcept:

		cdef int r, intRank, nRanks = <uint>len( INT_RANKS )
		cdef ll  product = 1

		# Check which bits are set to find ranks
		for r from 0 <= r < nRanks:
			intRank = INT_RANKS[ r ]
			if rankbits & (1 << r):
				product *= PRIMES[ r ]

		return product

	# Debugging function. Displays binary number as a human-readable str in groups of four digits
	@staticmethod
	cdef str  int_to_binary( int card_int ): #noexcept:

		cdef:
			str  bstr   = bin( card_int )[2:][::-1] # chop off the 0b and THEN reverse string
			list output = list( ''.join([ '0000' + '\t' ]*7) + '0000' ) 
			int  S      = <int>len( bstr ), s

		for s from 0 <= s < S:
			output[ s + s//4 ] = bstr[ s ]

		output.reverse()
		return ''.join( output )

	# Just gives us a nice formatted unicode string for a single card
	@staticmethod
	cdef str  int_to_pretty_str( int card_int, bint compact=FALSE, bint center=TRUE ): #noexcept:

		if card_int==0: 
			return '  '

		cdef int  suit_int = Card.get_suit_int( card_int ), rank_int = Card.get_rank_int( card_int )
		cdef bint red_suit = ((suit_int==2) or (suit_int==4))

		# Do we need to colour the suit red?
		cdef str s = PRETTY_SUITS[ suit_int ].decode(), r = STR_RANKS[ rank_int ]
		if red_suit: 
			s = colored(s, "red")

		cdef str cStr = f" [ {r}{s} ] " if not compact else f"{r}{s}"
		if center: 
			cStr = cStr.center(6) if not red_suit else '  ' + cStr + '  ' # Coloring fucks up .center()???
		return cStr 

	# Uses the above to get a formatted card string then prints it
	@staticmethod
	cdef void print_pretty_card( int card_int ): #noexcept:
		print( Card.int_to_pretty_str( card_int ) )

	# Uses the above to print formatted card strings for multiple cards
	@staticmethod
	cdef void print_pretty_cards( list card_ints ): #noexcept:

		cdef str output = ' '
		cdef int nCards = <int>len( card_ints ), i, c

		for i in range( nCards ):
			c = <int>card_ints[ i ]
			output += Card.int_to_pretty_str( c ) + (',' if i < nCards-1 else ' ')
			
		print( output )

# *-* #