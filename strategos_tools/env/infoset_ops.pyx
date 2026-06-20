# distutils: language = c
# cython: language_level 3
# cython: profile = False


cimport cython
from libc.string   cimport memcpy, memset
from cpython.array cimport clone as pyarr

cimport strategos_tools.env.card_ops as CardOps
from strategos_tools.env.player_ops cimport Are_Opponents, NextNonDealer, OpponentsOf
from strategos_tools.env.event_ops  cimport Is_Dealer_Action

from numpy import uintc, asarray as NP


# ==================================================================================================
# An infoset is sort of just a wrapper around a gamenode which implements the concept of hidden
# information. It reports gamestate information with information unobservable to the infoset's
# POV player redacted. Basically, we interact with an infoset just like we would interact with
# a gamenode, but it automatically redacts any information which is POV-unknowable. In the case
# of poker, this is just the private cards held by anyone other than the POV player.
# ==================================================================================================


cdef class infoset:

	def __init__( self, gamenode sourceNode, uint perspective_of ): 
		self.__INIT__( sourceNode, perspective_of )
		
	cdef void    __INIT__( self, gamenode sourceNode, uint perspective_of ): #noexcept:
		self.POVplayer = perspective_of
		self.OPPplayer = OpponentsOf( self.POVplayer )[0]
		self.hLen      = sourceNode.hLen
		self._n        = sourceNode

	# Returns an array describing the game's initial conditions.
	cdef uint1     InitialConditions( self ): #noexcept:
		return self._n.InitialConditions()

	# Returns last event from hist; start-point and type-filter options. Auto redacts hidden info.
	cdef gameevent LastEvent( self, uint from_point=0, uint of_type=NULLEVENT ): #noexcept:

		cdef gameevent lastEvent = self._n.LastEvent( from_point, of_type )

		# No hidden information to remove here
		if not lastEvent.DealTo == self.OPPplayer:
			return lastEvent 

		# By now we know that lastEvent is an opp deal, so redact the dealt cards
		cdef uint1 eArr = lastEvent.to_array()
		eArr[ CDEALT ]=0 

		return gameevent( from_array=eArr )

	# Just counts number of deal events that have occurred.
	cdef uint      NumDeals( self ): #noexcept:
		return self._n.NumDeals()

	# Returns array of all player IDs in game, regardless of player active state.
	cdef uint1     AllPlayers( self ): #noexcept:
		return self._n.AllPlayers()

	# Returns array of all players who are still able to act (i.e. are not folded or all-in).
	cdef uint1     ActivePlayers( self ): #noexcept:
		return self._n.ActivePlayers()

	# Returns ID of player expected to act now.
	cdef uint      ActingPlayer( self, uint at_point=0 ): #noexcept:
		return self._n.ActingPlayer( at_point )

	# Is I.POV the acting player?
	cdef bint      POV_Is_Acting_Player( self ): #noexcept:
		return self.POVplayer == self.ActingPlayer()

	# Sometimes useful to know how many players are active without needing to know who they are.
	cdef uint      NumActivePlayers( self ): #noexcept:
		return self._n.NumActivePlayers()

	# Returns array of the POV player's private card vecs.
	cdef uint2     HoleCards( self, bint Fill_To_Max=FALSE ): #noexcept:
		return self._n.HoleCards( self.POVplayer, Fill_To_Max )

	# Returns array of board card vecs.
	cdef uint2     BoardCards( self, bint Fill_To_Max=FALSE ): #noexcept:
		return self._n.BoardCards( Fill_To_Max )

	# Returns history index where the current betting round started.
	cdef uint      CurrentRoundStart( self ): #noexcept:
		return self._n.CurrentRoundStart()

	# Returns subhistory consisting of just the current betting round with any hidden info redacted.
	cdef uint2     CurrentRoundHist( self ): #noexcept:

		# If current round doesn't include opp deals, nothing to redact, just return true history
		if self.CurrentStreet() != PREFLOP:
			return self._n.CurrentRoundHist()

		cdef uint2 trueRoundHist = self._n.CurrentRoundHist(),                                                         \
				   subjRoundHist = cyarr( (trueRoundHist.shape[ 0 ], EVEC_SIZE), UINTSIZE, 'I' )
		cdef uint  roundLength   = subjRoundHist.shape[ 0 ], s

		# Since we know we're preflop now, make sure any opponent deals are redacted
		for s from 0 <= s < roundLength: 
			subjRoundHist[ s ] = trueRoundHist[ s ] 
			if trueRoundHist[ s,DEALTO ] == self.OPPplayer: 
				subjRoundHist[ s,CDEALT ] = 0

		return subjRoundHist

	# Returns current betting round ID.
	cdef uint      CurrentStreet( self ): #noexcept:
		return self._n.CurrentStreet()

	# Are we awaiting a player (i.e. nondealer) fold/call response to an all-in action?
	cdef bint      Awaiting_AllIn_Response( self ): #noexcept:
		return self._n.Awaiting_AllIn_Response()

	# Have all players who are able to act done so in the current round?
	cdef bint      Players_Have_Acted( self ): #noexcept:
		return self._n.Players_Have_Acted( players=self.ActivePlayers(), from_point=self.CurrentRoundStart() )

	# Calculates total each player has put into pot, option to start from a specific history step
	cdef uint1     BetTotals( self, uint from_point=1 ): #noexcept:
		return self._n.BetTotals( from_point )

	# Is the current betting round over?
	cdef bint      Current_Round_Over( self ): #noexcept:
		return self._n.Current_Round_Over()

	# Gets a specified player's current total stack, defaults to POV player.
	cdef uint      CurrentStack( self, uint for_player=0 ): #noexcept:
		return self._n.CurrentStacks()[ for_player if (for_player > 0) else self.POVplayer ]

	# Is the POV player posting a blind?
	cdef uint      Is_Posting_Blind( self ): #noexcept:

		cdef uint blindState = self._n.BlindState()
		
		if (blindState==0): 
			return FALSE

		elif (blindState==SB) and (self.POVplayer==self._n.SmallBlindPlayer): 
			return SB

		elif (blindState==BB) and (self.POVplayer==self._n.BigBlindPlayer):   
			return BB
			
		return FALSE

	# Indicates whether current game position is the first player action after blinds have been posted.
	cdef bint      Is_First_Action( self ): #noexcept:
		return self.hLen==self._n.BLINDS_DONE

	# Calculates POV player's min required bet (i.e. CALL) amount at the current game position.
	cdef uint      CallAmount( self ): #noexcept:

		# Find player with most in round's pot ⟶ find how much POVplayer needs to put in to match it
		cdef uint1 roundBetTotals = self.BetTotals( from_point=self.CurrentRoundStart() )
		cdef uint  POVBetTotal    = roundBetTotals[ self.POVplayer ], amtToMatch=0, p

		for p from 1 <= p <= self._n.PLAYER_COUNT:
			if roundBetTotals[ p ] > amtToMatch: 
				amtToMatch = roundBetTotals[ p ]
			
		return amtToMatch - POVBetTotal

	# Calculates the minimum raise the POV player is allowed to do.
	cdef uint      MinRaise( self ): #noexcept:

		# We just treat blinds as raises where the raise amt is determined by the blind being posted
		cdef uint Posting_Blind = self.Is_Posting_Blind()
		if Posting_Blind==SB: 
			return self._n.SmallBlindAmt
		if Posting_Blind==BB:
			return self._n.BigBlindAmt - self._n.SmallBlindAmt # This raiseAmt gives BetTotal = BBamt

		cdef:
			uint      currentRound    = self.CurrentStreet(),                                                          \
				      roundStart      = self._n.BLINDS_DONE if currentRound==PREFLOP else self.CurrentRoundStart()
			gameevent lastRoundRaise  = self.LastEvent( from_point=roundStart, of_type=RAISE )
			bint      First_Round_Bet = lastRoundRaise.Type==NULLEVENT

		# Typical logic: Must raise ≥ prev existing raise; if none exists, then min is BB
		return lastRoundRaise.RaiseAmt if (not First_Round_Bet) else self._n.BigBlindAmt

	# Calculates maximum POV player raise allowed by stack/blind limitations.
	cdef uint      MaxRaise( self ): #noexcept:

		# Blinds just treated as raises where the raise amt is determined by the blind being posted
		cdef uint Posting_Blind = self.Is_Posting_Blind()
		if Posting_Blind==SB: 
			return self._n.SmallBlindAmt
		if Posting_Blind==BB: 
			return self._n.BigBlindAmt - self._n.SmallBlindAmt # This raiseAmt gives BetTotal = BBamt

		# Not posting blind, so max raise = amount by which POV current stack exceeds call amt
		cdef uint minTotalBet = self.CallAmount(), currentStack = self.CurrentStack()
		return 0 if currentStack <= minTotalBet else currentStack - minTotalBet

	# Just gets current total pot size.
	cdef uint      PotSize( self ): #noexcept:
		return self._n.PotSize()

	# Did the previous action end the game?
	cdef bint      Is_Terminal( self ): #noexcept:
		return self._n.Is_Terminal()

	# Returns hist steps where cards were dealt to to_player; util for counterfactual stuff downstream.
	cdef uint1     DealSteps( self, uint to_player=ANY ): #noexcept:

		cdef uint  nDeals    = MAX_DEALT_CARDS if to_player==ANY else MAX_HOLE_CARDS, d=0, s
		cdef uint1 dealSteps = pyarr( ARR_TMPLT_I, nDeals, zero=True ), stepAction

		for s from 1 <= s < self.hLen:
			stepAction = self._n.History[ s ]

			if Is_Dealer_Action( stepAction ):
				if to_player==ANY or stepAction[ DEALTO ]==to_player:
					dealSteps[ d ] = s
					d+=1

			if d==nDeals:
				return dealSteps

		return dealSteps

	# Gets the unique ID for POV player's hand; util for ordering counterfactual histories downstream.
	cdef uint      HandIndex( self ): #noexcept:
		return CardOps.HandIndex( self.HoleCards() )

	# Returns arr of all cards currently observable to POVplayer (i.e. own hcards + any dealt bcards).
	cdef uint2     ObservableCards( self ): #noexcept:

		cdef:
			uint2 hCards   = self.HoleCards(), bCards = self.BoardCards()
			uint  nhCards  = hCards.shape[ 0 ], nbCards = bCards.shape[ 0 ], nCards = nhCards + nbCards, c
			uint2 allCards = cyarr( (nCards,CVEC_SIZE), UINTSIZE, 'I' )

		allCards[ :nhCards ] = hCards[:]
		allCards[ nhCards: ] = bCards[:]
		
		return allCards

	# Returns game history with dealt cards redacted from opponent deal events.
	cdef uint2     ObservableHistory( self ): #noexcept:

		cdef uint2 obHist = self._n.History.copy()
		cdef uint  stop   = self._n.PDEALS_DONE if (self.hLen >= self._n.PDEALS_DONE) else self.hLen, s

		for s from 0 <= s < stop: 
			if obHist[ s,DEALTO ]==self.OPPplayer:
				obHist[ s,CDEALT ] = 0 

		return obHist

	# Uses relations between current stack, call, and raise limits to calc number of possible raises.
	cdef uint    __NumAvailableRaises( self, uint currentStack, uint currentCall ): #noexcept:

		# poor
		if currentStack < currentCall:
			return 0

		cdef uint minRaise = self.MinRaise(), minRaiseTotal = currentCall + minRaise

		# only available raise is an all-in
		if currentCall < currentStack <= minRaiseTotal:
			return 1

		cdef uint maxRaise = self.MaxRaise()

		# typical case where stack exceeds min raise
		if currentStack > minRaiseTotal:
			return (maxRaise - minRaise) + 1

	# Calculates the total number of actions available to POVplayer at the current game position.
	cdef uint      NumAvailableActions( self ): #noexcept:
	
		if not self.POV_Is_Acting_Player(): 
			return 0

		cdef bint Posting_Blind = self.Is_Posting_Blind() > 0
		if Posting_Blind:
			return 1

		cdef uint stack   = self.CurrentStack(),                                                                       \
				  callAmt = self.CallAmount(),                                                                         \
				  nRaises = self.__NumAvailableRaises( stack, callAmt )

		# Now determine which nonraise (FOLD, CHECK, CALL) actions are available
		cdef bint Can_Fold  = (not Posting_Blind),                                                                     \
				  Can_Check = (not Posting_Blind) and (callAmt==0) and (self._n.CurrentStreet()!=PREFLOP),             \
				  Can_Call  = (not Posting_Blind) and (not Can_Check) and stack > 0

		return nRaises + (<uint>Can_Fold + <uint>Can_Check + <uint>Can_Call)

	# Returns an array containing all possible hands the opponent could be holding.
	# TODO: Generalize to >2pl, will require adding an arg specifying which opponent
	cdef uint3     PossibleOppHands( self ): #noexcept:

		# First get deck with all POV-observable cards removed
		cdef:
			uint2 deck     = CardOps.FilteredDeck( excludeCards=self.ObservableCards() )
			uint3 oppHands = cyarr( (NUM_POSSIBLE_HANDS, MAX_HOLE_CARDS, CVEC_SIZE), UINTSIZE, 'I' )
			uint  deckSize = deck.shape[ 0 ], h=0, c1, c2

		# Now collect every possible 2-card combination from filtered deck
		for c1 from 0 <= c1 < deckSize-1:
			for c2 from c1 < c2 < deckSize:
				oppHands[ h,0 ] = deck[ c1 ]
				oppHands[ h,1 ] = deck[ c2 ]
				h+=1

		return oppHands[ :h ]

	# Does the same thing as PossibleOppHands except returns array of hand inds instead of cardVecs.
	# TODO: Generalize to >2pl, will require adding an arg specifying which opponent
	cdef uint1     PossibleOppHandInds( self ): #noexcept:

		cdef:
			uint3 oppHands = self.PossibleOppHands()
			uint1 hInds    = pyarr( ARR_TMPLT_I, NUM_POSSIBLE_HANDS, zero=False )
			uint  nHands   = oppHands.shape[ 0 ], h
			uint2 oppHand

		for h from 0 <= h < nHands:
			oppHand  = oppHands[ h ] # (2,CVEC_SIZE)
			hInds[h] = CardOps.HandIndex( oppHand )

		return hInds[ :h ]

	# Returns arr containing all possible opp-perspective histories corresponding to possible opp hands.
	# TODO: Generalize to >2pl - array will be huge since num of combinations for >2 opponents will be massive
	cdef uint3     PossibleOppHistories( self ): #noexcept:

		# First get deal positions and possible opp hands
		cdef:
			uint1 oppDeals  = self.DealSteps( to_player=self.OPPplayer ),                                              \
				  povDeals  = self.DealSteps( to_player=self.POVplayer )
			uint2 obHist    = self.ObservableHistory()
			uint3 oppHands  = self.PossibleOppHands(),                                                                 \
				  possibleH = cyarr( (oppHands.shape[0], self.hLen, EVEC_SIZE), UINTSIZE, 'I' )
			uint  pDeal1 = povDeals[ 0 ], pDeal2 = povDeals[ 1 ], oDeal1 = oppDeals[ 0 ], oDeal2 = oppDeals[ 1 ]

		possibleH[...] = obHist # Create nHands copies of observed history
		if pDeal1 > 0: possibleH[:, pDeal1, CDEALT] = 0 # POVplayer cards are redacted to opp
		if pDeal2 > 0: possibleH[:, pDeal2, CDEALT] = 0 # POVplayer cards are redacted to opp
		if oDeal1 > 0: possibleH[:, oDeal1, CDEALT] = oppHands[:,0,CARD] # Insert opp cards at deal steps
		if oDeal2 > 0: possibleH[:, oDeal2, CDEALT] = oppHands[:,1,CARD] # Insert opp cards at deal steps

		return possibleH

	# Outputs a pydict containing state information; used for serializing game trajectory data.
	cdef dict      to_dict( self ): #noexcept:

		cdef uint1 iConds = self._n.InitialConditions().copy()
		cdef uint2 H      = self._n.History.copy()
		return {
			'POV':       self.POVplayer,
			'InitConds': NP( iConds,dtype=uintc ),
			'gHist':     NP( H,dtype=uintc )
		}

	# Just dumps a bunch of info about the current infoset for human-readable gamestate inspection.
	cdef void      summary( self, bint Append_To_Node_Summary=FALSE ): #noexcept:

		cdef:
			uint      rStart      = self.CurrentRoundStart(), blind = self.Is_Posting_Blind(), s, c
			uint1     bTotals     = self.BetTotals(), rbTotals = self.BetTotals( from_point=rStart ), eArr
			uint2     bCards      = self.BoardCards(), hCards = self.HoleCards(), oCards = self.ObservableCards(),     \
					  subjHist    = self.ObservableHistory()
			tuple     BLINDSTATES = ( "NOPE", "SB", "BB" )
			str       bPretty     = ''.join( CardOps.PrettyCardStrings( bCards ) ),                                    \
				      hPretty     = ''.join( CardOps.PrettyCardStrings( hCards ) ),                                    \
				      oPretty     = ''.join( CardOps.PrettyCardStrings( oCards ) ),                                    \
				      blindStr    = BLINDSTATES[ blind ]
			gameevent lRaise      = self.LastEvent( from_point=rStart,of_type=RAISE ), stepEvent


		if not Append_To_Node_Summary:
			print( '\n'+('='*100 ))
			print( f"P{self.POVplayer} INFOSET SUMMARY".center(100) )
			print( '='*100 )		

			print( '\t' + ('='*50) )
			print( '\t' + f"STATE INFO".center(50) )
			print( '\t' + ('='*50) )
			print( '\t' + f"hLen".rjust(20)                 + f" || {self.hLen}" )
			print( '\t' + f"Current Round".rjust(20)        + f" || {self.CurrentStreet()}" )
			print( '\t' + f"Acting Player".rjust(20)        + f" || {PLAYERNAMES[ self.ActingPlayer() ]}" )
			print( '\t' + f"POV Is Posting Blind".rjust(20) + f" || {'YES' if blind>0 else 'NO'}"+blindStr )
			print( '\t' + f"POV Game Bet Total".rjust(20)   + f" || {bTotals[ self.POVplayer ]}" )
			print( '\t' + f"POV Round Bet Total".rjust(20)  + f" || {rbTotals[ self.POVplayer ]}" )
			print( '\t' + f"POV Hole Cards".rjust(20)       + f" || {hPretty}" )
			print( '\t' + f"Board Cards".rjust(20)          + f" || {bPretty}" )
			print( '\t' + f"Observable Cards".rjust(20)     + f" || {oPretty}" )
			print()

			print( '\t' + ('='*50) )
			print( '\t' + f"ACTION CONSTRAINTS".center(50) )
			print( '\t' + ('='*50) )
			print( '\t' + f"Last Round Raise".rjust(20)    + f" || {lRaise.ShortString() if lRaise.Type==RAISE else 'NONE'}" )
			print( '\t' + f"Current Pot Size".rjust(20)    + f" || {self.PotSize()}" )
			print( '\t' + f"POV Stack".rjust(20)           + f" || {self.CurrentStack()}" )
			print( '\t' + f"Current Call Amount".rjust(20) + f" || {self.CallAmount()}" )
			print( '\t' + f"POV Min Raise".rjust(20)       + f" || {self.MinRaise()}" )
			print( '\t' + f"POV Max Raise".rjust(20)       + f" || {self.MaxRaise()}" )
			print( '\t' + f"|A_pov|".rjust(20)             + f" || {self.NumAvailableActions()}" )
			print()

			print( f"\tLastEvent = {self.LastEvent().ShortString()}" )
			print( '\t' + ('='*85) )
			print( '\t' + f"P{self.POVplayer} OBSERVABLE HISTORY".center(85) )
			print( '\t' + ('='*85) )
			print( '\t' + f" STEP | | TYPE | |PLAYER| | RAMT | | BTOT | |ALL_IN| |ALLIND| |DEALTO| |CDEALT| | " )
			for s from 0 <= s <= self.hLen:
				if s < self.hLen:
					stepEvent = gameevent( from_array=subjHist[ s ] )
					print( f"\t{ stepEvent.ShorterString( s ) }" )
				elif not self.Is_Terminal():
					print( f"\t{str( s ).rjust(6)}| |" + f"++ CURRENT STEP - ACTION NOT YET CHOSEN ++".center(74) )
			print()

		if Append_To_Node_Summary:
			print( f"\n\tAS OBSERVED BY PLAYER {PLAYERNAMES[ self.POVplayer ]}:" )
			for s from 0 <= s < self.hLen:
				stepEvent = gameevent( from_array=subjHist[ s ] )
				print( f"\t{stepEvent.ShorterString(s)}" )


	# Reconstructs an infoset object from a saved pydict.
	@staticmethod
	cdef infoset from_dict( dict Idict ): #noexcept:

		cdef:
			uint  POV       = Idict[ 'POV' ]
			uint1 initConds = Idict[ 'InitConds' ]
			uint2 gHist     = Idict[ 'gHist' ]
			gamenode recoveredNode = gamenode( initialConditions=initConds, history=gHist )

		return infoset( sourceNode=recoveredNode, perspective_of=POV )
		
		
# *-* # 