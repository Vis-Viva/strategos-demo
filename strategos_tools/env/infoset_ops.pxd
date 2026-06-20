#distutils: language = c
#cython: language_level 3


cimport cython
from cython.view cimport array as cyarr

from strategos_tools.core.CONSTS cimport *
from strategos_tools.env.event_ops cimport gameevent
from strategos_tools.env.gamenode_ops cimport gamenode


cdef class infoset:

	cdef:
		uint     POVplayer, OPPplayer, hLen
		gamenode _n

	cdef void    __INIT__( self, gamenode sourceNode, uint perspective_of ) #noexcept

	cdef uint1     InitialConditions( self ) #noexcept

	cdef gameevent LastEvent( self, uint from_point=*, uint of_type=* ) #noexcept

	cdef uint      NumDeals( self ) #noexcept

	cdef uint1     AllPlayers( self ) #noexcept

	cdef uint1     ActivePlayers( self ) #noexcept

	cdef uint      ActingPlayer( self, uint at_point=* ) #noexcept

	cdef bint      POV_Is_Acting_Player( self ) #noexcept

	cdef uint      NumActivePlayers( self ) #noexcept

	cdef uint2     HoleCards( self, bint Fill_To_Max=* ) #noexcept

	cdef uint2     BoardCards( self, bint Fill_To_Max=* ) #noexcept

	cdef uint      CurrentRoundStart( self ) #noexcept

	cdef uint2     CurrentRoundHist( self ) #noexcept

	cdef uint      CurrentStreet( self ) #noexcept

	cdef bint      Awaiting_AllIn_Response( self ) #noexcept

	cdef bint      Players_Have_Acted( self ) #noexcept

	cdef uint1     BetTotals( self, uint from_point=* ) #noexcept

	cdef bint      Current_Round_Over( self ) #noexcept

	cdef uint      CurrentStack( self, uint for_player=* ) #noexcept

	cdef uint      Is_Posting_Blind( self ) #noexcept

	cdef bint      Is_First_Action( self ) #noexcept

	cdef uint      CallAmount( self ) #noexcept

	cdef uint      MinRaise( self ) #noexcept

	cdef uint      MaxRaise( self ) #noexcept

	cdef uint      PotSize( self ) #noexcept

	cdef bint      Is_Terminal( self ) #noexcept

	cdef uint1     DealSteps( self, uint to_player=* ) #noexcept

	cdef uint      HandIndex( self ) #noexcept

	cdef uint2     ObservableCards( self ) #noexcept

	cdef uint2     ObservableHistory( self ) #noexcept

	cdef uint    __NumAvailableRaises( self, uint currentStack, uint currentCall ) #noexcept

	cdef uint      NumAvailableActions( self ) #noexcept
	
	cdef uint3     PossibleOppHands( self ) #noexcept

	cdef uint1     PossibleOppHandInds( self ) #noexcept

	cdef uint3     PossibleOppHistories( self ) #noexcept

	cdef dict      to_dict( self ) #noexcept

	cdef void      summary( self, bint Append_To_Node_Summary=* ) #noexcept

	@staticmethod
	cdef infoset   from_dict( dict Idict ) #noexcept


# *-* #
