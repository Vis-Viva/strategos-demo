#distutils: language = c
#cython: language_level 3

from strategos_tools.core.CONSTS cimport *

cdef str STR_RANKS, INT_SUIT_TO_CHAR_SUIT
cdef tuple INT_RANKS, PRIMES
cdef dict CHAR_RANK_TO_INT_RANK, CHAR_SUIT_TO_INT_SUIT
cdef dict PRETTY_SUITS

cdef class Card:

	@staticmethod
	cdef int  new_card( str card_str ) #noexcept

	@staticmethod
	cdef str  int_to_str( int card_int ) #noexcept

	@staticmethod
	cdef int  get_rank_int( int card_int ) #noexcept

	@staticmethod
	cdef int  get_suit_int( int card_int ) #noexcept

	@staticmethod
	cdef int  get_bitrank_int( int card_int ) #noexcept

	@staticmethod
	cdef int  get_prime( int card_int ) #noexcept

	@staticmethod
	cdef list hand_to_binary( list card_strs ) #noexcept

	@staticmethod
	cdef ll   prime_product_from_hand( int[::1] card_ints ) #noexcept

	@staticmethod
	cdef ll   prime_product_from_rankbits( int rankbits ) #noexcept

	@staticmethod
	cdef str  int_to_binary( int card_int ) #noexcept

	@staticmethod
	cdef str  int_to_pretty_str( int card_int, bint compact=*, bint center=* ) #noexcept

	@staticmethod
	cdef void print_pretty_card( int card_int ) #noexcept

	@staticmethod
	cdef void print_pretty_cards( list card_ints ) #noexcept
