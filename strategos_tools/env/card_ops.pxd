#distutils: language = c
#cython: language_level 3


from strategos_deuces.evaluator cimport Evaluator as HandEvaluator
from strategos_tools.core.CONSTS cimport *
from strategos_tools.core.containers cimport *


cdef index_map __HandIdxMap() #noexcept

cdef:
	index_map     HAND_IDX_MAP
	HandEvaluator EVALUATOR
	

cdef bint  Card_Is_Null( uint1 cardVec ) #noexcept

cdef uint  CardID( uint rankID, uint suitID ) #noexcept

cdef uint1 CardVector( uint cardID ) #noexcept

cdef int   DeuceSuitIndex( uint1 cardVec ) #noexcept

cdef int   DeuceRankIndex( uint1 cardVec ) #noexcept

cdef int   DeuceRankPrime( uint1 cardVec ) #noexcept

cdef int1  DeuceInts( uint2 cardVecs ) #noexcept

cdef uint2 VecDeck() #noexcept

cdef uint2 FilteredDeck( uint2 excludeCards, bint Include_Gaps=* ) #noexcept

cdef uint1 Draw( uint2 from_deck ) #noexcept

cdef uint2 CombineCards( uint2 holeCards, uint2 boardCards ) #noexcept

cdef uint1 CombineInts( uint1 holeInts, uint1 boardInts ) #noexcept

cdef uint  HandEval( uint2 holeCards, uint2 boardCards ) #noexcept

cdef ll    HandProduct( uint2 hand ) #noexcept

cdef uint  HandIndex( uint2 hand ) #noexcept

cdef uint1 card_vector_from_deuceint( int deuceInt ) #noexcept

cdef uint2 cvecs_from_deuceints( int1 deuceInts ) #noexcept

cdef uint1 card_vector_from_str( str cardString ) #noexcept

cdef uint2 card_vectors_from_strings( list cardStrings ) #noexcept

cdef int   card_int_from_str( str cardString ) #noexcept

cdef uint2 BestHand( uint2 holeCards, uint2 boardCards ) #noexcept

cdef str   RankClass( int hScore ) #noexcept

cdef list  RankSuitStrings( uint2 cardVecs ) #noexcept

cdef uint2 find_high_card( uint2 holeCards ) #noexcept

cdef uint2 find_pairs( uint2 fullHand, uint2 holeCards ) #noexcept

cdef uint2 find_three_of_a_kind( uint2 fullHand, uint2 holeCards ) #noexcept

cdef uint2 find_four_of_a_kind( uint2 fullHand, uint2 holeCards ) #noexcept

cdef uint2 find_winning_cards( uint2 from_winning_hand, uint2 with_hole_cards, str of_rank_class ) #noexcept

cdef list  PrettyCardStrings( uint2 cardVecs, bint Compact=*, bint Center=* ) #noexcept


# *-* #