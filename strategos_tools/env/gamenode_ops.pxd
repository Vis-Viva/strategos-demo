#distutils: language = c
#cython: language_level 3


cimport cython

cimport numpy as cnp
cnp.import_array()

from strategos_tools.core.CONSTS   cimport *
from strategos_tools.env.event_ops cimport gameevent


cdef class gamenode:

	cdef:
		uint2 History
		uint1 InitialStacks
		uint  PLAYER_COUNT, hLen, ButtonPos, SmallBlindPlayer, BigBlindPlayer, SmallBlindAmt, BigBlindAmt,             \
			  NUM_PDEALS, PDEALS_DONE, BLINDS_DONE, SB_STEP, BB_STEP

	cdef void    __AUTOINIT__( self, uint2 history, uint1 initialConditions ) #noexcept

	cdef void    __MANUAL_INIT__( self, uint2 history, uint nPlayers, uint bpos, uint smallBlind, uint1 initStacks ) #noexcept

	cdef uint1     InitialConditions( self ) #noexcept

	cdef gameevent LastEvent( self, uint from_point=*, uint of_type=* ) #noexcept

	cdef uint2     HoleCards( self, uint for_player, bint Fill_To_Max=* ) #noexcept

	cdef uint2     BoardCards( self, bint Fill_To_Max=* ) #noexcept

	cdef uint2     AllDealtCards( self, bint Validate=* ) #noexcept

	cdef uint2     AvailableDeck( self, bint Include_Gaps=*, bint Validate=* ) #noexcept

	cdef uint      NumDeals( self, uint to_player=*, bint Include_AllIn=* ) #noexcept

	cdef uint      NumFolds( self ) #noexcept

	cdef uint      NumAllIns( self ) #noexcept

	cdef uint1     BetTotals( self, uint from_point=* ) #noexcept

	cdef uint1     AllPlayers( self ) #noexcept

	cdef uint1     ActivePlayers( self ) #noexcept

	cdef uint      NumActivePlayers( self ) #noexcept

	cdef uint      CurrentRoundStart( self ) #noexcept

	cdef uint1     StackAdjustments( self ) #noexcept

	cdef uint1     CurrentStacks( self ) #noexcept

	cdef uint      CurrentStreet( self ) #noexcept

	cdef uint      BlindState( self ) #noexcept

	cdef bint      Betting_Has_Started( self ) #noexcept

	cdef uint2     AllInSequence( self ) #noexcept

	cdef bint      Awaiting_AllIn_Deal( self ) #noexcept

	cdef bint      Awaiting_AllIn_Response( self ) #noexcept

	cdef bint      Round_Deals_Done( self ) #noexcept

	cdef bint      Players_Have_Acted( self, uint1 players, uint from_point=*, bint Include_Blinds=* ) #noexcept

	cdef bint      All_Bets_Match( self, uint1 for_players, uint from_point=* ) #noexcept

	cdef bint      Current_Round_Over( self ) #noexcept

	cdef bint      Is_Terminal( self ) #noexcept

	cdef gamenode  Predecessor( self, uint with_hlen=* ) #noexcept

	cdef bint      Is_Dealer_Position( self ) #noexcept

	cdef bint      At_Round_Start( self ) #noexcept

	cdef uint      ActingPlayer( self, uint at_point=* ) #noexcept

	cdef gamenode  ParentNode( self, bint Include_Deals=* ) #noexcept

	cdef uint2     CurrentRoundHist( self ) #noexcept

	cdef uint      PotSize( self ) #noexcept

	cdef uint1     HandScores( self ) #noexcept

	cdef uint1     Winners( self, uint1 from_scores ) #noexcept

	cdef int1      ShowdownResults( self, uint potSize, uint1 betTotals, uint1 stackAdjustments ) #noexcept

	cdef int1      GameResults( self ) #noexcept

	cdef uint2     CurrentBestHand( self, uint for_player ) #noexcept

	cdef uint      LastCompletedPDeal( self ) #noexcept

	cdef uint      DealTo( self ) #noexcept

	cdef gameevent Deal( self ) #noexcept

	cdef uint2     SubHistory( self, uint of_length ) #noexcept

	cdef gamenode  Successor( self, gameevent e ) #noexcept

	cdef uint      NumDecisionPoints( self, uint for_player=* ) #noexcept

	cdef ll        GTKey_old( self ) #noexcept

	cdef ll        GTKey( self ) #noexcept

	cdef void      summary( self, bint Compact=* ) #noexcept

	cdef void      DIAGNOSTIC( self ) #noexcept


cdef gamenode RootNode( uint1 initialConditions=*, uint players=*, uint buttonPos=*, uint smallBlind=*, uint1 initStacks=* ) #noexcept

cdef gamenode DummyNode() #noexcept


# *-* #