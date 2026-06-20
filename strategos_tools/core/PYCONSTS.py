from sys   import getsizeof as sizeof
from numpy import asarray as NP, empty, uintc
from PIL   import Image as IMG


# ----- UNIVERSAL CONSTS ---------------------------------------------------------------------------


LOGO = "\n\n\
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

SEG_ADV_DIR = "data/segadvs/" 
SEG_REC_DIR = "data/segrecs/" 
LINE_UP     = '\033[1A'
LINE_CLEAR  = '\x1b[2K'

INTSIZE  = sizeof( int )
FLTSIZE  = sizeof( float )


# ----- PLAYER CONSTS ------------------------------------------------------------------------------


NUM_PLAYERS = 2 # Eventually won't be able to use this as a const, instead pref gamenode.PLAYER_COUNT
DEALER      = 0
INITIAL_POV = 1
ANY         = 420 # Indicates "any player" for ops involving player filtering
ALLPLAYERS  = NP( range( NUM_PLAYERS+1 ),dtype=uintc )
PLAYERCODES = [ "D" ]      + [ f"P{n}" for n in range( 1,NUM_PLAYERS+1 ) ]
PLAYERNAMES = [ "DEALER" ] + [ f"P{n}" for n in range( 1,NUM_PLAYERS+1 ) ]


# ----- CARD CONSTS --------------------------------------------------------------------------------


MAX_STRAIGHT_FLUSH  = 10
MAX_FOUR_OF_A_KIND  = 166
MAX_FULL_HOUSE      = 322
MAX_FLUSH           = 1599
MAX_STRAIGHT        = 1609
MAX_THREE_OF_A_KIND = 2467
MAX_TWO_PAIR        = 3325
MAX_PAIR            = 6185
MAX_HIGH_CARD       = 7462

DEUCE_RANK_CHARS = 'x23456789TJQKA'
DEUCE_SUIT_CHARS = 'xshdc'
DEUCE_PRIMES     = NP( (2,3,5,7,11,13,17,19,23,29,31,37,41),dtype=uintc )

NULLRANK = 0
NULLSUIT = 0
SPADES   = 1
HEARTS   = 2
DIAMONDS = 3
CLUBS    = 4

BOARD                = 0
HOLE                 = 1
MAX_HOLE_CARDS       = 2
MAX_BOARD_CARDS      = 5
MAX_HAND_SIZE        = 5
NUM_RANKS            = 13
NUM_SUITS            = 4
NUM_CARD_SETS        = 4 # hole, flop, turn, river
DECK_SIZE            = 52
LARGEST_PRIME        = 239
NUM_POSSIBLE_HANDS   = 1326 # 52 choose 2
MAX_DEALT_CARDS      = ( MAX_HOLE_CARDS * NUM_PLAYERS ) + MAX_BOARD_CARDS # =9 (for headsup)
MAX_OBSERVABLE_CARDS = MAX_HOLE_CARDS + MAX_BOARD_CARDS # =7

# Card vec indexing/sizing 
CARD      = 0
RANK      = 1
SUIT      = 2
CVEC_SIZE = 3 # [cID,rID,sID]

NULLCARD      = 0 # NULLCARD ID, & FULL_VEC_DECK idx where null card vector [0 0 0] lives
FULL_VEC_DECK = empty( (DECK_SIZE+1,CVEC_SIZE), dtype=uintc )
CARD_STRINGS  = []

FULL_VEC_DECK[ NULLCARD,CARD ] = NULLCARD
FULL_VEC_DECK[ NULLCARD,RANK ] = NULLRANK
FULL_VEC_DECK[ NULLCARD,SUIT ] = NULLSUIT
for _r in range( 1,NUM_RANKS+1 ):
     for _s in range( 1,NUM_SUITS+1 ):

          _c = (((_r-1)*NUM_SUITS) + (_s-1))+1
          FULL_VEC_DECK[ _c,CARD ] = _c
          FULL_VEC_DECK[ _c,RANK ] = _r
          FULL_VEC_DECK[ _c,SUIT ] = _s

          _cStr = DEUCE_RANK_CHARS[ _r ] + DEUCE_SUIT_CHARS[ _s ]
          CARD_STRINGS.append( _cStr )


# ----- EVENT CONSTS -------------------------------------------------------------------------------


# Event type identification
NULLEVENT         = 0
FOLD              = 1
CHECK             = 2
CALL              = 3
RAISE             = 4
BOARDDEAL         = 5
PLAYERDEAL        = 6
NUM_ETYPES        = 7
NUM_PLAYER_ETYPES = 4
NON_RAISE_ETYPES  = 3
TYPECODES         = ( "N", "F", "CH", "CA", "R", "BD", "PD" )
TYPENAMES         = ( "NULL ", "FOLD ", "CHECK", "CALL ", "RAISE", "BDEAL", "PDEAL" )
EKEYS = ( "EventType", "PlayedBy", "RaiseAmount", "TotalBet", "Is_AllIn", "AllInDiff", "DealtTo", "CardsDealt" )
	 
# Event vector indexing/sizing
TYPE      = 0
PLAYEDBY  = 1
IS_ALLIN  = 2
BETTOTAL  = 3
RAISEAMT  = 4
ALLINDIFF = 5
DEALTO    = 6
CDEALT    = 7
EVEC_SIZE = 8


# ----- NODE CONSTS --------------------------------------------------------------------------------


# Initial condition array indexing/sizing
NPLR       = 0
BPOS       = 1
SBAM       = 2
STK        = 3
STK1       = 4
STK2       = 5
NUM_ICONDS = 6

# Round identifiers
PREFLOP    = 0
FLOP       = 1
TURN       = 2
RIVER      = 3
NUM_ROUNDS = 4
ROUNDNAMES = [ "PREFLOP", "FLOP", "TURN", "RIVER" ] 

FLOP_DEAL_SIZE  = 3
TURN_DEAL_SIZE  = 1
RIVER_DEAL_SIZE = 1

# Blind IDs
SB         = 1
BB         = 2
NUM_BLINDS = 2


# ----- NN CONSTS ----------------------------------------------------------------------------------


ADVNET_INPUT_LEN = 15 # total number of input tensors for AdvNet.forward()
BASE_BATCH_SIZE  = 32768
MAX_BATCH_SIZE   = 131072
TRAIN_EPOCHS     = 4096
BASE_LRATE       = 0.001
MAX_GPUS         = 8
SET_DIM          = 1 # axis for summing embeddings over sets of permutation-invariant cards
FEATURE_DIM      = 1 # axis for concatenating various encodings during fwd pass


# *-* #
