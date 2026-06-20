# distutils: language = c
# cython: language_level 3
# cython: profile = False

cimport cython
from libc.stdlib   cimport rand as RNG
from libc.string   cimport memcpy
from cpython.array cimport clone as pyarr
from cython.view   cimport array as cyarr

from strategos_deuces.card cimport Card as Deuce

import numpy as np

from numpy     import asarray as NP, log2, uintc
from termcolor import colored


# ==================================================================================================
# We represent cards in different ways for different purposes. By default, we use a single int ∈ 
# [1-52]. When we ultimately use cards as neural net inputs later, we use card vectors of the form:
# [cardID, rankID, suitID], cardID ∈ [1,52], rankID ∈ [1,13], suitID ∈ [1,4]. This module provides
# basic card-level ops such as drawing from the deck, ops for moving between card representations,
# tools for interfacing with the Deuces module to do hand scoring, and some more esoteric stuff 
# like assigning unique IDs to all 2-card hands which we later use to order potential hand histories.
# ==================================================================================================


# Assigns a unique int ID to every 2-card hand, allows us to order counterfactual histories later
cdef index_map       __HandIdxMap(): #noexcept:

	cdef:
		index_map handIdxMap = index_map()
		uint2     hand
		int1      hInts
		ll        hProd
		uint      hIdx=0, c1, c2

	for c1 from 1 <= c1 <= DECK_SIZE-1:
		for c2 from c1 < c2 <= DECK_SIZE:

			hand    = cyarr( (MAX_HOLE_CARDS,CVEC_SIZE), UINTSIZE, 'I' )
			hand[0] = FULL_VEC_DECK[ c1 ]
			hand[1] = FULL_VEC_DECK[ c2 ]

			hInts = DeuceInts( hand )
			hProd = hInts[ 0 ] * hInts[ 1 ]

			handIdxMap.append( hProd,hIdx )
			hIdx+=1
			
	return handIdxMap

# Instantiate module-level singletons
cdef index_map     HAND_IDX_MAP = __HandIdxMap()
cdef HandEvaluator EVALUATOR    = HandEvaluator() # hand evaluator from Deuces


cdef inline bint  Card_Is_Null( uint1 cardVec ): #noexcept:
	return cardVec[ CARD ]==NULLCARD

cdef inline uint  CardID( uint rankID, uint suitID ): #noexcept:
	return (((rankID-1)*NUM_SUITS) + (suitID-1))+1

cdef uint1 CardVector( uint cardID ): #noexcept:
	return FULL_VEC_DECK[ cardID ]

# Deuce suit idx = 2^sIdx, where sIdx is the suit's position in the string 'shdc'
cdef int   DeuceSuitIndex( uint1 cardVec ): #noexcept:
	cdef int sIdx = <int>(cardVec[ SUIT ]-1)
	return <int>(2**sIdx)

# Same deal but for ranks, where the deuce idx is just rIdx instead of 2^rIdx
cdef int   DeuceRankIndex( uint1 cardVec ): #noexcept:
	return <int>(cardVec[ RANK ]-1)

cdef int   DeuceRankPrime( uint1 cardVec ): #noexcept:
	return <int>(DEUCE_PRIMES[ DeuceRankIndex( cardVec ) ])

# ∀ cVec ∈ cardVecs: find deuce suit & rank inds ⟶ replicate deuce's bitmath to get cVec's deuce int
cdef int1  DeuceInts( uint2 cardVecs ): #noexcept:

	cdef:
		uint1 cVec
		int   nCards    = cardVecs.shape[ 0 ], c, s_int, r_int, rank_prime, bitrank, suit, rank, deuceInt
		int1  deuceInts = pyarr( ARR_TMPLT_i, nCards, zero=False )

	for c from 0 <= c < nCards:
		cVec = cardVecs[ c ]

		if Card_Is_Null( cVec ): deuceInts[ c ]=0
		else:
			s_int      = DeuceSuitIndex( cVec ) 
			r_int      = DeuceRankIndex( cVec )
			rank_prime = DeuceRankPrime( cVec )

			# Deuce's bitmath
			bitrank        = 1 << r_int << 16
			suit           = s_int << 12
			rank           = r_int << 8
			deuceInt       = bitrank | suit | rank | rank_prime
			deuceInts[ c ] = deuceInt

	return deuceInts

# Just guarantees that we don't interfere with FULL_VEC_DECK itself (and filters out NULLCARD)
cdef uint2 VecDeck(): #noexcept:
	return FULL_VEC_DECK.copy()[1:]

# Return FULL_VEC_DECK minus anything in excludeCards. Gaps are just for pretty printing
cdef uint2 FilteredDeck( uint2 excludeCards, bint Include_Gaps=FALSE ): #noexcept:

	cdef uint  nxCards  = excludeCards.shape[ 0 ]
	cdef uint2 fullDeck = VecDeck()
	
	if nxCards==0:
		return fullDeck 

	cdef:
		uint  nCards   = DECK_SIZE if Include_Gaps else DECK_SIZE-nxCards, n=0, d, c, deckCardID
		uint2 partDeck = cyarr( (nCards,CVEC_SIZE), UINTSIZE, 'I' )
		uint1 deckCard
		bint  Card_Excluded

	if Include_Gaps: partDeck[:]=0
	for d from 0 <= d < DECK_SIZE:
		deckCard      = fullDeck[ d ]
		deckCardID    = deckCard[ CARD ]
		Card_Excluded = FALSE

		for c from 0 <= c < nxCards: # Determine whether card is excluded
			if excludeCards[ c,CARD ]==deckCardID: 
				Card_Excluded=TRUE

		if not Card_Excluded:
			partDeck[ n ] = deckCard
			n+=1
		elif Include_Gaps: # Skip an idx, so partDeck[skippedIdx,:]=0 - useful for printing decks
			n+=1 

	return partDeck

# Draws a cardVec from the deck and returns it, stupid
cdef uint1 Draw( uint2 from_deck ): #noexcept:
	cdef uint randIdx = (RNG()) % (from_deck.shape[0])
	return from_deck[ randIdx ]

# Combines provided set arrays of card vecs into a single array
cdef uint2 CombineCards( uint2 holeCards, uint2 boardCards ): #noexcept:

	cdef uint  h = holeCards.shape[ 0 ], b = boardCards.shape[ 0 ], nCards = h+b
	cdef uint2 combinedCards = cyarr( (nCards,CVEC_SIZE), UINTSIZE, 'I' )

	combinedCards[ :h ] = holeCards
	combinedCards[ h: ] = boardCards
	# SLICING REPLACEMENT
	#memcpy( &combinedCards[ 0,0 ], &holeCards[ 0,0 ],  h*CVEC_BYTES )
	#memcpy( &combinedCards[ h,0 ], &boardCards[ 0,0 ], b*CVEC_BYTES )
	return combinedCards

# Combines provided set arrays of card ints into a single array 
cdef uint1 CombineInts( uint1 holeInts, uint1 boardInts ): #noexcept:

	cdef uint  h = holeInts.shape[ 0 ], b = boardInts.shape[ 0 ], nCards = h+b
	cdef uint1 combinedInts = pyarr( ARR_TMPLT_I, nCards, zero=False )

	combinedInts[ :h ] = holeInts
	combinedInts[ h: ] = boardInts
	# SLICING REPLACEMENT
	#memcpy( &combinedInts[ 0 ], &holeInts[ 0 ],  h*UINTSIZE )
	#memcpy( &combinedInts[ h ], &boardInts[ 0 ], b*UINTSIZE )
	return combinedInts

# Convert card vecs into deuce ints ⟶ feed them into the cythonized deuce evaluator
cdef uint  HandEval( uint2 holeCards, uint2 boardCards ): #noexcept:

	cdef uint2 combinedCards = CombineCards( holeCards,boardCards )
	cdef int1 deuceCards     = DeuceInts( combinedCards )
	return EVALUATOR.evaluate( deuceCards )

# Gives us the unique product of Deuce ints for the specified hand
cdef ll    HandProduct( uint2 hand ): #noexcept:
	cdef int1 deuceHand = DeuceInts( hand )
	return <ll>(deuceHand[ 0 ] * deuceHand[ 1 ])

# Assigning a unique idx to every 2-card hand allows us to consistently align histories downstream based on dealt cards
cdef uint  HandIndex( uint2 hand ): #noexcept:
	return HAND_IDX_MAP.at( HandProduct( hand ) )

cdef uint1 card_vector_from_deuceint( int deuceInt ): #noexcept:
	
	cdef uint1 cVec = pyarr( ARR_TMPLT_I, CVEC_SIZE, zero=True )
	if deuceInt==0: return cVec

	cdef int  dRank = Deuce.get_rank_int( deuceInt ), dSuit = Deuce.get_suit_int( deuceInt )
	cdef uint rID  = <uint>dRank,                     sID  = <uint>log2( dSuit )
	
	cVec[ CARD ] = CardID( rID,sID ); cVec[ RANK ] = rID; cVec[ SUIT ] = sID
	return cVec

cdef uint2 cvecs_from_deuceints( int1 deuceInts ): #noexcept:
	
	cdef uint  nCards = deuceInts.size, c
	cdef uint2 cVecs  = cyarr( (nCards,CVEC_SIZE), UINTSIZE, 'I' )
	for c from 0 <= c < nCards: cVecs[ c ] = card_vector_from_deuceint( deuceInts[ c ] )
	return cVecs

# Assumes cardString is a valid 'Rs'-type string. Responsibility to ensure this is on the caller.
cdef uint1 card_vector_from_str( str cardString ): #noexcept:
	
	cdef:
		uint1 cVec = pyarr( ARR_TMPLT_I, CVEC_SIZE, zero=True )
		str   rank = cardString[ 0 ], suit = cardString[ 1 ]
		uint  rID  = DEUCE_RANK_CHARS.find( rank ), sID = DEUCE_SUIT_CHARS.find( suit )

	cVec[ CARD ] = CardID( rID,sID ); cVec[ RANK ] = rID; cVec[ SUIT ] = sID
	return cVec

# Assumes cardStrings are valid 'Rs'-type strings. Responsibility to ensure this is on the caller.
cdef uint2 card_vectors_from_strings( list cardStrings ): #noexcept:
	
	cdef: 
		str   cStr
		uint  nCards = <uint>len( cardStrings ), c
		uint2 cVecs  = cyarr( (nCards,CVEC_SIZE), UINTSIZE, 'I' )

	for c from 0 <= c < nCards:
		cStr       = cardStrings[ c ]
		cVecs[ c ] = card_vector_from_str( cStr )

	return cVecs

# Converts 'Rs' strings to deuce integers
cdef int   card_int_from_str( str cardString ): #noexcept:
	
	cdef uint2 cVec = cyarr( (1,CVEC_SIZE), UINTSIZE, 'I' ) # This has to be 2d so we can use it as input to DeuceInts
	cVec[ 0 ] = card_vector_from_str( cardString )
	return DeuceInts( cVec )[0]

cdef uint2 BestHand( uint2 holeCards, uint2 boardCards ): #noexcept:

	cdef:
		int   deuceInt
		uint2 cardVecs   = CombineCards( holeCards, boardCards )
		int1  deuceCards = DeuceInts( cardVecs ), bestHand = EVALUATOR.get_best_hand( deuceCards )
		list  cStrings   = [ Deuce.int_to_str( deuceInt ) for deuceInt in bestHand ]

	return card_vectors_from_strings( cStrings )

cdef str   RankClass( int hScore ): #noexcept:
	return EVALUATOR.class_to_string( EVALUATOR.get_rank_class( hScore ) )

cdef list  RankSuitStrings( uint2 cardVecs ): #noexcept:

	cdef:
		uint  nCards = cardVecs.shape[ 0 ], c, rID, sID
		uint1 cVec
		str   rStr, sStr
		list  cStrings = []

	for c from 0 <= c < nCards:
		cVec = cardVecs[ c ]
		rID  = cVec[ RANK ]; rStr = DEUCE_RANK_CHARS[ rID ]
		sID  = cVec[ SUIT ]; sStr = DEUCE_SUIT_CHARS[ sID ]
		cStrings.append( rStr+sStr )

	return cStrings

cdef uint2 find_high_card( uint2 holeCards ): #noexcept:
	
	cdef:
		uint  nCards = holeCards.shape[ 0 ], highRank=0, c, rID
		uint1 cVec
		uint2 highCard = cyarr( (1,CVEC_SIZE), UINTSIZE, 'I' ) # Yeah it's dumb, but this has to be 2d because reasons

	for c from 0 <= c < nCards:
		cVec = holeCards[ c ]
		rID  = cVec[ RANK ]
		if rID >= highRank:
			highRank    = rID
			highCard[0] = cVec

	return highCard

cdef uint2 find_pairs( uint2 fullHand, uint2 holeCards ): #noexcept:
	
	cdef:
		str  cStr, rank
		list handStrings = RankSuitStrings( fullHand ),         holeStrings = RankSuitStrings( holeCards ),            \
			 handRanks   = [ cStr[0] for cStr in handStrings ], holeRanks   = [ cStr[0] for cStr in holeStrings ],     \
			 pairs=[]
		uint c, handSize = <uint>len( handStrings )

	for c from 0 <= c < handSize:
		rank = handRanks[ c ]; cStr = handStrings[ c ]
		if handRanks.count( rank )>=2: pairs.append( cStr )

	return card_vectors_from_strings( pairs )

cdef uint2 find_three_of_a_kind( uint2 fullHand, uint2 holeCards ): #noexcept:

	cdef:
		str  cStr
		list handStrings = RankSuitStrings( fullHand ),  handRanks = [ cStr[0] for cStr in handStrings ],              \
			 holeStrings = RankSuitStrings( holeCards ), holeRanks = [ cStr[0] for cStr in holeStrings ],              \
			 threeOfAKind=[]
		uint handSize    = <uint>len( handStrings ), c

	for c from 0 <= c < handSize:
		if handRanks.count( handRanks[c] )>=3: threeOfAKind.append( handStrings[c] ) 

	return card_vectors_from_strings( threeOfAKind )

cdef uint2 find_four_of_a_kind( uint2 fullHand, uint2 holeCards ): #noexcept:
	
	cdef:
		str  cStr
		list handStrings = RankSuitStrings( fullHand ),  handRanks = [ cStr[0] for cStr in handStrings ],              \
			 holeStrings = RankSuitStrings( holeCards ), holeRanks = [ cStr[0] for cStr in holeStrings ],              \
			 fourOfAKind = []
		uint handSize    = <uint>len( handStrings ), c

	for c from 0 <= c < handSize:
		if handRanks.count( handRanks[c] )>=4: fourOfAKind.append( handStrings[c] )

	return card_vectors_from_strings( fourOfAKind )

# BestHand above gives us the 5-card winning hand, from which we now extract the subset which produced the winning score 
cdef uint2 find_winning_cards( uint2 from_winning_hand, uint2 with_hole_cards, str of_rank_class ): #noexcept:

	if of_rank_class=="High Card":       return find_high_card( with_hole_cards )
	if of_rank_class=="Pair":            return find_pairs( from_winning_hand, with_hole_cards )
	if of_rank_class=="Two Pair":        return find_pairs( from_winning_hand, with_hole_cards )
	if of_rank_class=="Three of a Kind": return find_three_of_a_kind( from_winning_hand, with_hole_cards )
	if of_rank_class=="Four of a Kind":  return find_four_of_a_kind( from_winning_hand, with_hole_cards )
	else: return from_winning_hand
	# ^All other rank classes are 5card hands. from_winning_hand is 5 cards by definition, so just return to sender

# Just takes some cardVecs, converts to deuce integers, then uses deuce's card formatting to make it pretty *-* 
cdef list  PrettyCardStrings( uint2 cardVecs, bint Compact=FALSE, bint Center=TRUE ): #noexcept:

	cdef uint nCards   = cardVecs.shape[ 0 ]
	cdef bint No_Cards = nCards==0 or (NP( cardVecs )==0).all()
	if No_Cards: return [' '*6]

	cdef int1 deuceInts = DeuceInts( cardVecs )
	cdef int  c
	return [ Deuce.int_to_pretty_str( c, Compact, Center ) for c in deuceInts ]


# ----- PYTHON INTERFACE FUNCTIONS -----------------------------------------------------------------


def cvecs_from_strings( cStrings ): 
	return NP( card_vectors_from_strings( cStrings ),dtype=uintc )
	
def pretty_card_strings( cardVecs, Compact=TRUE, Center=FALSE ): 
	return PrettyCardStrings( cardVecs, Compact, Center )


# *-* # 