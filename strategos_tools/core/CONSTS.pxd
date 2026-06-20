#distutils: language = c
#cython: language_level 3

cimport cython
from cpython.array cimport array as cparr

cimport numpy as cnp
cnp.import_array()



# ----- Type alises -------------------------------------------------------------------------------
ctypedef long long          ll
ctypedef unsigned int       uint
ctypedef unsigned char      uchar
ctypedef unsigned long long ull
ctypedef cnp.ndarray        nparr

# ----- C-contiguous ndim 32 bit ℕ arrays ----------------------------------------------------------
ctypedef uint[::1]       uint1
ctypedef uint[:,::1]     uint2
ctypedef uint[:,:,::1]   uint3
ctypedef uint[:,:,:,::1] uint4

# ----- C-contiguous ndim 32 bit ℤ arrays ----------------------------------------------------------
ctypedef int[::1]       int1
ctypedef int[:,::1]     int2
ctypedef int[:,:,::1]   int3
ctypedef int[:,:,:,::1] int4

# ----- C-contiguous ndim 128 bit ℤ arrays ---------------------------------------------------------
ctypedef ll[::1]       ll1
ctypedef ll[:,::1]     ll2
ctypedef ll[:,:,::1]   ll3
ctypedef ll[:,:,:,::1] ll4

# ----- C-contiguous ndim 32 bit ℝ arrays ----------------------------------------------------------
ctypedef float[::1]       flt1
ctypedef float[:,::1]     flt2
ctypedef float[:,:,::1]   flt3
ctypedef float[:,:,:,::1] flt4

# ----- C-contiguous ndim 64 bit ℝ arrays ----------------------------------------------------------
ctypedef double[::1]       dbl1
ctypedef double[:,::1]     dbl2
ctypedef double[:,:,::1]   dbl3
ctypedef double[:,:,:,::1] dbl4


# ----- UNIVERSAL CONSTS ---------------------------------------------------------------------------


cdef:
	str   LOGO, LINE_UP, LINE_CLEAR, SEG_ADV_DIR, SEG_REC_DIR
	uint  UINTSIZE, INTSIZE, LLSIZE, FLTSIZE, DBLSIZE, PTRSIZE
	cparr ARR_TMPLT_I, ARR_TMPLT_i, ARR_TMPLT_ll, ARR_TMPLT_f, ARR_TMPLT_d
	bint  TRUE, FALSE


# ----- PLAYER CONSTS ------------------------------------------------------------------------------


cdef:
	list  PLAYERCODES
	list  PLAYERNAMES
	uint  NUM_PLAYERS
	uint  DEALER
	uint  INITIAL_POV
	uint  ANY
	uint1 ALLPLAYERS 


# ----- CARD CONSTS --------------------------------------------------------------------------------


cdef:
	uint  MAX_STRAIGHT_FLUSH
	uint  MAX_FOUR_OF_A_KIND
	uint  MAX_FULL_HOUSE
	uint  MAX_FLUSH
	uint  MAX_STRAIGHT
	uint  MAX_THREE_OF_A_KIND
	uint  MAX_TWO_PAIR
	uint  MAX_PAIR
	uint  MAX_HIGH_CARD
	str   DEUCE_RANK_CHARS
	str   DEUCE_SUIT_CHARS
	uint1 DEUCE_PRIMES

	uint NULLRANK
	uint NULLSUIT
	uint SPADES
	uint HEARTS
	uint DIAMONDS
	uint CLUBS

	uint BOARD
	uint HOLE
	uint MAX_HOLE_CARDS
	uint MAX_BOARD_CARDS
	uint MAX_HAND_SIZE
	uint NUM_RANKS
	uint NUM_SUITS
	uint DECK_SIZE
	uint LARGEST_PRIME
	uint NUM_POSSIBLE_HANDS
	uint MAX_DEALT_CARDS
	uint MAX_OBSERVABLE_CARDS
	
	uint CARD
	uint RANK
	uint SUIT
	uint CVEC_SIZE

	uint  NULLCARD
	uint2 FULL_VEC_DECK
	list  CARD_STRINGS


# ----- EVENT CONSTS -------------------------------------------------------------------------------


cdef:
	uint  NULLEVENT
	uint  FOLD
	uint  CHECK
	uint  CALL
	uint  RAISE
	uint  BOARDDEAL
	uint  PLAYERDEAL
	uint  NUM_ETYPES
	uint  NUM_PLAYER_ETYPES
	uint  NON_RAISE_ETYPES
	tuple TYPECODES
	tuple TYPENAMES
	tuple EKEYS
	
	uint TYPE
	uint PLAYEDBY
	uint RAISEAMT
	uint BETTOTAL
	uint IS_ALLIN
	uint ALLINDIFF
	uint DEALTO
	uint CDEALT
	uint EVEC_SIZE


# ----- NODE CONSTS --------------------------------------------------------------------------------


cdef:
	uint NPLR
	uint BPOS
	uint SBAM
	uint STK 
	uint STK1
	uint STK2
	uint NUM_ICONDS

	uint PREFLOP
	uint FLOP
	uint TURN
	uint RIVER
	uint NUM_ROUNDS
	list ROUNDNAMES

	uint FLOP_DEAL_SIZE
	uint TURN_DEAL_SIZE
	uint RIVER_DEAL_SIZE

	uint SB
	uint BB
	uint NUM_BLINDS


# ----- NN CONSTS ----------------------------------------------------------------------------------


cdef:
	uint  BASE_BATCH_SIZE
	uint  MAX_BATCH_SIZE
	uint  TRAIN_EPOCHS
	float BASE_LRATE
	uint  MAX_GPUS

	
# *-* #