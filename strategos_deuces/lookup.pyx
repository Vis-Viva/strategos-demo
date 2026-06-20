#distutils: language = c
#cython: language_level 3
#cython: profile = False

from cpython.array cimport clone as pyarr
from cython.view cimport array as cyarr
from strategos_deuces.card cimport *

import itertools


# ==================================================================================================
# Cythonized version of Deuce's lookup table for hand scoring. Extremely fast and lightweight. 
# Strategos does not really interface with this directly; it's just used for Deuce's hand evaluator.
# The lookup table maps each 5-card hand's unique prime product to a rank in the range [1, 7462].
#
# Number of distinct hand values:
# 	Straight Flush   10 
# 	Four of a Kind   156      [(13 choose 2) * (2 choose 1)]
# 	Full Houses      156      [(13 choose 2) * (2 choose 1)]
# 	Flush            1277     [(13 choose 5) - 10 straight flushes]
# 	Straight         10 
# 	Three of a Kind  858      [(13 choose 3) * (3 choose 1)]
# 	Two Pair         858      [(13 choose 3) * (3 choose 2)]
# 	One Pair         2860     [(13 choose 4) * (4 choose 1)]
# 	High Card      + 1277     [(13 choose 5) - 10 straights]
# 	-------------------------
# 	TOTAL            7462
#
# For example: 
# 	- Royal flush (best hand possible)         = 1
#	- 7-5-4-3-2 unsuited (worst hand possible) = 7462
# ==================================================================================================


# Module consts
cdef dict MAX_TO_RANK_CLASS = {	MAX_STRAIGHT_FLUSH:  1,
								MAX_FOUR_OF_A_KIND:  2,
								MAX_FULL_HOUSE:      3,
								MAX_FLUSH:           4,
								MAX_STRAIGHT:        5,
								MAX_THREE_OF_A_KIND: 6,
								MAX_TWO_PAIR:        7,
								MAX_PAIR:            8,
								MAX_HIGH_CARD:       9 }

cdef dict RANK_CLASS_TO_STRING = { 1 : "Straight Flush",
								   2 : "Four of a Kind",
								   3 : "Full House",
								   4 : "Flush",
								   5 : "Straight",
								   6 : "Three of a Kind",
								   7 : "Two Pair",
								   8 : "Pair",
								   9 : "High Card" }


cdef class LookupTable:

	def __init__( self ):
		self.flush_lookup    = {}
		self.unsuited_lookup = {}
		self.flushes()
		self.multiples()

	# Sets up hand score ranks for flushes
	cdef void flushes( self ): #noexcept:
		"""
		Straight flushes and flushes. 

		Lookup is done on 13 bit integer (2^13 > 7462):
		xxxbbbbb bbbbbbbb => integer hand index
		"""

		cdef tuple straightFlushes = ( 7936,  # = int('0b1111100000000', 2), royal flush
									   3968,  # = int('0b111110000000', 2),
									   1984,  # = int('0b11111000000', 2),
									   992,   # = int('0b1111100000', 2),
									   496,   # = int('0b111110000', 2),
									   248,   # = int('0b11111000', 2),
									   124,   # = int('0b1111100', 2),
									   62,    # = int('0b111110', 2),
									   31,    # = int('0b11111', 2),
									   4111 ) # = int('0b1000000001111', 2), 5 high
		cdef int NUM_STRAIGHT_FLUSHES = <int>len( straightFlushes )

		cdef:
			list   flushes = []
			object gen     = self.bit_seq_gen( int('0b11111',2) )
			int    f, sf, flushBits, straightFlushBits
			bint   unstraight_flush

		for f from 0 <= f < 1277 + NUM_STRAIGHT_FLUSHES - 1:
			flushBits = next( gen )
			unstraight_flush = True

			for sf from 0 <= sf < NUM_STRAIGHT_FLUSHES:
				straightFlushBits = straightFlushes[ sf ]

				if not (flushBits ^ straightFlushBits):
					unstraight_flush = False

			if unstraight_flush: 
				flushes.append( flushBits )

		flushes.reverse()

		cdef int NUM_FLUSHES = <int>len( flushes ), rank=1
		cdef ll  primeProd

		for sf from 0 <= sf < NUM_STRAIGHT_FLUSHES:
			straightFlushBits = straightFlushes[ sf ]
			primeProd         = C.prime_product_from_rankbits( straightFlushBits )

			self.flush_lookup[ primeProd ] = rank
			rank+=1

		rank = (<int>MAX_FULL_HOUSE)+1
		for f from 0 <= f < NUM_FLUSHES:
			flushBits = flushes[ f ]
			primeProd = C.prime_product_from_rankbits( flushBits )

			self.flush_lookup[ primeProd ] = rank
			rank+=1

		self.straight_and_highcards( straightFlushes, flushes )

	cdef void straight_and_highcards( self, tuple straights, list highcards ): #noexcept:

		cdef int rank = (<int>MAX_FLUSH)+1, s, h
		cdef ll  primeProd

		for s in straights:
			primeProd = C.prime_product_from_rankbits( s )
			self.unsuited_lookup[ primeProd ] = rank
			rank+=1

		rank = (<int>MAX_PAIR)+1
		for h in highcards:
			primeProd = C.prime_product_from_rankbits( h )
			self.unsuited_lookup[ primeProd ] = rank
			rank+=1

	cdef void multiples( self ): #noexcept:
		
		cdef:
			list backwards_ranks = list( range( len( INT_RANKS )-1, -1, -1 ) ), kickers
			int  rank = (<int>MAX_STRAIGHT_FLUSH)+1, i, k, pr, r
			ll   primeProd

		for i in backwards_ranks:
			kickers = backwards_ranks[:]
			kickers.remove( i )
			
			for k in kickers:
				primeProd = (PRIMES[ i ]**4) * PRIMES[ k ]
				self.unsuited_lookup[ primeProd ] = rank
				rank+=1
		
		rank = (<int>MAX_FOUR_OF_A_KIND)+1
		cdef list pairranks
		for i in backwards_ranks:
			pairranks = backwards_ranks[:]
			pairranks.remove( i )

			for pr in pairranks:
				primeProd = (PRIMES[ i ]**3) * (PRIMES[ pr ]**2)
				self.unsuited_lookup[ primeProd ] = rank
				rank+=1

		cdef:
			object gen
			int    c1,c2
			tuple  kicker_pair

		rank = (<int>MAX_STRAIGHT)+1
		for r in backwards_ranks:
			kickers = backwards_ranks[:]
			kickers.remove( r )
			gen = itertools.combinations( kickers,2 )

			for kicker_pair in gen:
				c1,c2     = kicker_pair
				primeProd = PRIMES[ r ]**3 * PRIMES[ c1 ] * PRIMES[ c2 ]

				self.unsuited_lookup[ primeProd ] = rank
				rank+=1

		rank = (<int>MAX_THREE_OF_A_KIND)+1
		gen  = itertools.combinations( backwards_ranks,2 )
		cdef tuple tp

		for tp in gen:
			c1,c2   = tp
			kickers = backwards_ranks[:]
			kickers.remove( c1 )
			kickers.remove( c2 )

			for k in kickers:
				primeProd = PRIMES[ c1 ]**2 * PRIMES[ c2 ]**2 * PRIMES[ k ]
				self.unsuited_lookup[ primeProd ] = rank
				rank+=1

		cdef int   pairrank, k1, k2, k3
		cdef tuple kicker

		rank = (<int>MAX_TWO_PAIR)+1
		for pairrank in backwards_ranks:
			kickers = backwards_ranks[:]
			kickers.remove( pairrank )
			gen = itertools.combinations( kickers,3 )

			for kicker in gen:
				k1,k2,k3  = kicker
				primeProd = (PRIMES[ pairrank ]**2) * PRIMES[ k1 ] * PRIMES[ k2 ] * PRIMES[ k3 ]
				self.unsuited_lookup[ primeProd ] = rank
				rank+=1

	cdef void write_table_to_disk( self, LookupTable table, str filepath ): #noexcept:
		with open( filepath,'w' ) as f:
			for primeProd, rank in table.items():
				f.write( str( primeProd ) + ',' + str( rank ) + '\n' )

	# Bit hack from http://www-graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
	# Generator does this is poker order rank, so no need to sort when done.
	def bit_seq_gen( self, int bits ) -> int:
		cdef int t = (bits | (bits-1)) + 1, next = t | ((((t & -t) // (bits & -bits)) >> 1) - 1)
		yield next
		while True:
			t = (next | (next - 1)) + 1; next = t | ((((t & -t) // (next & -next)) >> 1) - 1)
			yield next

# *-* #