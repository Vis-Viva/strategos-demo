# distutils: language = c
# cython: language_level 3
# cython: profile = False

cimport cython
cimport numpy as cnp
cnp.import_array()

from cpython.array cimport array as cparr
from cython.view   cimport array as cyarr

from numpy import asarray as NP, uintc


# ==================================================================================================
# Various named constant quantities useful throughout strategos's code. These all live in the same
# place so it's easy to make sure we always import what we need. Nothing here is particularly
# heavyweight anyway, so importing it all even when we only need part of it isn't a big deal.
# ==================================================================================================


# ----- UNIVERSAL CONSTS ---------------------------------------------------------------------------


cdef:
	str LOGO = "\n\n\
	███████████████████████████████████████████═══╗\n\
	██ ╔═════════════════════════════════════██═╗ ║\n\
	██ ║                                     ██ ║ ║\n\
	██ ║   ████████████████████████████      ██ ║ ║\n\
	██ ║   █████▄════════════════════██═╗    ██ ║ ║\n\
	██ ║    ▀█████▄═══════════════════█ ║    ██ ║ ║\n\
	██ ║      ▀█████▄                 ▝╗║    ██ ║ ║\n\
	██ ║        ▀█████▄                ╚╝    ██ ║ ║\n\
	██ ║          ▀█████▄                    ██ ║ ║\n\
	██ ║            ▀█████▄                  ██ ║ ║\n\
	██ ║              ▀███▀═╗                ██ ║ ║\n\
	██ ║             ▄██▀ ╔═╝                ██ ║ ║\n\
	██ ║           ▄██▀ ╔═╝                  ██ ║ ║\n\
	██ ║         ▄██▀ ╔═╝                    ██ ║ ║\n\
	██ ║       ▄██▀ ╔═╝                 ▗    ██ ║ ║\n\
	██ ║     ▄██▀ ╔═╝                   █╔╗  ██ ║ ║\n\
	██ ║   ▄██▀ ╔═╝                    ██╝║  ██ ║ ║\n\
	██ ║  ███████████████████████████████ ║  ██ ║ ║\n\
	██ ║  ███████████████████████████████ ║  ██ ║ ║\n\
	██ ║   ╚══════════════════════════════╝  ██ ║ ║\n\
	██ ║                                     ██ ║ ║\n\
	███████████████████████████████████████████ ║ ║\n\
	 ║ ╚════════════════════════════════════════╝ ║\n\
	 ╚════════════════════════════════════════════╝\n\
	\n\
	████████╗ █████████╗ ████████╗ ████████╗ █████████╗ ████████╗ ████████╗ ████████╗ ████████╗\n\
	██╔═════╝  ╚═▐█▌╔══╝ ██╔═══██║ ██╔═══██║  ╚═▐█▌╔══╝ ██╔═════╝ ██╔═════╝ ██╔═══██║ ██╔═════╝\n\
	██║          ▐█▌║    ██║   ██║ ██║   ██║    ▐█▌║    ██║       ██║       ██║   ██║ ██║\n\
	████████╗    ▐█▌║    ████████║ ████████║    ▐█▌║    █████╗    ██║ ████╗ ██║   ██║ ████████╗\n\
	╚═════██║    ▐█▌║    ██╔═██╔═╝ ██╔═══██║    ▐█▌║    ██╔══╝    ██║ ╚═██║ ██║   ██║ ╚═════██║\n\
	      ██║    ▐█▌║    ██║ ╚██╗  ██║   ██║    ▐█▌║    ██║       ██║   ██║ ██║   ██║       ██║\n\
	████████║    ▐█▌║    ██║  ╚██╗ ██║   ██║    ▐█▌║    ████████╗ ████████║ ████████║ ████████║\n\
	╚═══════╝     ╚═╝    ╚═╝   ╚═╝ ╚═╝   ╚═╝     ╚═╝    ╚═══════╝ ╚═══════╝ ╚═══════╝ ╚═══════╝\n\
\n\n"

	str SEG_ADV_DIR = "data/segadvs/" 
	str SEG_REC_DIR = "data/segrecs/" 
	str LINE_UP     = '\033[1A'
	str LINE_CLEAR  = '\x1b[2K'

	uint UINTSIZE = sizeof( uint )
	uint INTSIZE  = sizeof( int )
	uint LLSIZE   = sizeof( ll )
	uint FLTSIZE  = sizeof( float )
	uint DBLSIZE  = sizeof( double )
	uint PTRSIZE  = sizeof( void* )

	bint FALSE = 0
	bint TRUE  = 1

	cparr ARR_TMPLT_d  = cparr( 'd' ) # double array clone template
	cparr ARR_TMPLT_f  = cparr( 'f' ) # float array clone template
	cparr ARR_TMPLT_I  = cparr( 'I' ) # uint array clone template
	cparr ARR_TMPLT_i  = cparr( 'i' ) # int array clone template 
	cparr ARR_TMPLT_ll = cparr( 'q' ) # ll array clone template


# ----- PLAYER CONSTS ------------------------------------------------------------------------------


cdef:
	uint  NUM_PLAYERS = 2 # Eventually won't be able to use this as a const, instead pref gamenode.PLAYER_COUNT
	uint  DEALER      = 0
	uint  INITIAL_POV = 1
	uint  ANY         = 420 # Indicates "any player" for ops involving player filtering
	uint1 ALLPLAYERS  = NP( range( NUM_PLAYERS+1 ),dtype=uintc )
	list  PLAYERCODES = [ "D" ]      + [ f"P{n}" for n in range( 1,NUM_PLAYERS+1 ) ]
	list  PLAYERNAMES = [ "DEALER" ] + [ f"P{n}" for n in range( 1,NUM_PLAYERS+1 ) ]


# ----- CARD CONSTS --------------------------------------------------------------------------------


cdef:
	# Hand scoring values
	uint MAX_STRAIGHT_FLUSH  = 10
	uint MAX_FOUR_OF_A_KIND  = 166
	uint MAX_FULL_HOUSE      = 322
	uint MAX_FLUSH           = 1599
	uint MAX_STRAIGHT        = 1609
	uint MAX_THREE_OF_A_KIND = 2467
	uint MAX_TWO_PAIR        = 3325
	uint MAX_PAIR            = 6185
	uint MAX_HIGH_CARD       = 7462

	str   DEUCE_RANK_CHARS = 'x23456789TJQKA'
	str   DEUCE_SUIT_CHARS = 'xshdc'
	uint1 DEUCE_PRIMES     = NP( (2,3,5,7,11,13,17,19,23,29,31,37,41),dtype=uintc )

	# RankIDs etc
	uint NULLRANK = 0
	uint NULLSUIT = 0
	uint SPADES   = 1
	uint HEARTS   = 2
	uint DIAMONDS = 3
	uint CLUBS    = 4

	uint BOARD                = 0 # Index to access board cards inside card arrays
	uint HOLE                 = 1 # Index to access hole cards inside card arrays
	uint MAX_HOLE_CARDS       = 2
	uint MAX_BOARD_CARDS      = 5
	uint MAX_OBSERVABLE_CARDS = MAX_HOLE_CARDS + MAX_BOARD_CARDS # =7
	uint MAX_DEALT_CARDS      = ( MAX_HOLE_CARDS * NUM_PLAYERS ) + MAX_BOARD_CARDS # =9 (for headsup)
	uint MAX_HAND_SIZE        = 5
	uint NUM_RANKS            = 13
	uint NUM_SUITS            = 4
	uint DECK_SIZE            = 52
	uint LARGEST_PRIME        = 239 # Used for deuce hand evaluation logic
	uint NUM_POSSIBLE_HANDS   = 1326 # 52 choose 2

	# Card vec indexing/sizing 
	uint CARD      = 0
	uint RANK      = 1
	uint SUIT      = 2
	uint CVEC_SIZE = 3 # [cID,rID,sID]

	uint  NULLCARD      = 0 # NULLCARD ID & VEC_DECK idx where null card vector [0 0 0] lives
	uint2 FULL_VEC_DECK = cyarr( (DECK_SIZE+1,CVEC_SIZE), UINTSIZE, 'I' )
	list  CARD_STRINGS  = []

# Build the full deck of card vectors
cdef:
	str   _cStr
	uint  _c, _r, _s

FULL_VEC_DECK[ NULLCARD,CARD ] = NULLCARD
FULL_VEC_DECK[ NULLCARD,RANK ] = NULLRANK
FULL_VEC_DECK[ NULLCARD,SUIT ] = NULLSUIT

for _r from 1 <= _r <= NUM_RANKS:
	for _s from 1 <= _s <= NUM_SUITS:

		_c = ( ((_r-1)*NUM_SUITS) + (_s-1) ) + 1
		FULL_VEC_DECK[ _c,CARD ] = _c
		FULL_VEC_DECK[ _c,RANK ] = _r
		FULL_VEC_DECK[ _c,SUIT ] = _s

		_cStr = DEUCE_RANK_CHARS[ _r ] + DEUCE_SUIT_CHARS[ _s ]
		CARD_STRINGS.append( _cStr )


# ----- EVENT CONSTS -------------------------------------------------------------------------------


cdef:
	# Event type identification
	uint NULLEVENT         = 0
	uint FOLD              = 1
	uint CHECK             = 2
	uint CALL              = 3
	uint RAISE             = 4
	uint BOARDDEAL         = 5
	uint PLAYERDEAL        = 6
	uint NUM_ETYPES        = 7
	uint NUM_PLAYER_ETYPES = 4
	uint NON_RAISE_ETYPES  = 3

	# Some useful stuff for printing human-readable event details
	tuple TYPECODES = ( "N", "F", "CH", "CA", "R", "BD", "PD" )
	tuple TYPENAMES = ( "NULL ", "FOLD ", "CHECK", "CALL ", "RAISE", "BDEAL", "PDEAL" )
	tuple EKEYS     = ( "EventType", "PlayedBy", "RaiseAmount", "TotalBet", 
					"Is_AllIn", "AllInDiff", "DealtTo", "CardsDealt" )
	
	# Event vector indexing/sizing
	uint TYPE      = 0
	uint PLAYEDBY  = 1
	uint IS_ALLIN  = 2
	uint BETTOTAL  = 3
	uint RAISEAMT  = 4
	uint ALLINDIFF = 5
	uint DEALTO    = 6
	uint CDEALT    = 7
	uint EVEC_SIZE = 8


# ----- NODE CONSTS --------------------------------------------------------------------------------


cdef:
	# Initial condition array indexing/sizing
	uint NPLR       = 0 # Num players
	uint BPOS       = 1 # Button position
	uint SBAM       = 2 # Small blind amount
	uint STK        = 3 # Do initialConditions[ STK: ] for array of all players' starting stacks
	uint STK1       = 4 # Player1 stack index
	uint STK2       = 5 # Player2 stack index
	uint NUM_ICONDS = 6

	# Round identifiers
	uint PREFLOP    = 0
	uint FLOP       = 1
	uint TURN       = 2
	uint RIVER      = 3
	uint NUM_ROUNDS = 4
	list ROUNDNAMES = [ "PREFLOP", "FLOP", "TURN", "RIVER" ] 

	uint FLOP_DEAL_SIZE  = 3
	uint TURN_DEAL_SIZE  = 1
	uint RIVER_DEAL_SIZE = 1

	# Blind IDs
	uint SB         = 1
	uint BB         = 2
	uint NUM_BLINDS = 2


# ----- NN CONSTS ----------------------------------------------------------------------------------


cdef:
	uint  BASE_BATCH_SIZE = 32768
	uint  MAX_BATCH_SIZE  = 131072
	uint  TRAIN_EPOCHS    = 4096
	float BASE_LRATE      = 0.001
	uint  MAX_GPUS        = 8

	
# *-* #
