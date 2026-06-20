# distutils: language = c
# cython: language_level 3
# cython: profile = False


cimport cython

from libc.string   cimport memcpy, memset
from cpython.array cimport clone as pyarr
from cython.view   cimport array as cyarr

cimport strategos_tools.env.card_ops as CardOps

from strategos_tools.env.player_ops cimport NextNonDealer, OpponentsOf
from strategos_tools.env.event_ops  cimport Is_Dealer_Action, Is_Null

from numpy import asarray as NP, uintc


# ==================================================================================================
# A gamenode basically represents a HUNLTH gamestate. A "gamestate" is synonymous with a set of
# initial conditions (i.e. game parameters) and a game history consisting of a sequence of player
# actions which we call gameevents. The initial conditions + game history are the only actual
# information we store on a node. Instead of persisting tons of information like current stacks,
# current cards etc, we call functions to scan over history and compute this information as we
# need it in real time - this ends up being significantly faster than persisting all these game 
# variables and computing new updated values for them every time a new action evolves the state.
# The key takeaway is: gamenode = game state = initial conditions + history.
# NOTE: All game histories always start with a NULLEVENT vector of all 0s.
# ==================================================================================================


cdef class gamenode:

	# Initial conditions can either be specified individually, or provided in a special array
	def __init__( self, uint2 history, uint nPlayers=2, uint buttonPos=0, uint smallBlindAmt=0, uint1 initStacks=None, 
						uint1 initialConditions=None ):

		if initialConditions is None:
			self.__MANUAL_INIT__( history, nPlayers, buttonPos, smallBlindAmt, initStacks )

		else:
			self.__AUTOINIT__( history, initialConditions )

	# Initializes from an initial conditions array & player/dealer action history
	cdef void      __AUTOINIT__( self, uint2 history, uint1 initialConditions ): #noexcept:

		self.History          = history
		self.PLAYER_COUNT     = initialConditions[ NPLR ]
		self.ButtonPos        = initialConditions[ BPOS ]
		self.SmallBlindAmt    = initialConditions[ SBAM ]
		self.InitialStacks    = initialConditions[ STK: ] # [ 0,p1Stack,p2Stack ]
		self.SmallBlindPlayer = self.ButtonPos if self.PLAYER_COUNT==2 else NextNonDealer( self.ButtonPos )
		self.BigBlindPlayer   = NextNonDealer( self.SmallBlindPlayer )
		self.BigBlindAmt      = self.SmallBlindAmt*2 # TODO: GENERALIZE THIS
		self.hLen             = self.History.shape[ 0 ]

		# Useful early-game history indices; tells us first hist step AFTER certain stuff has been done
		self.NUM_PDEALS  = self.PLAYER_COUNT * MAX_HOLE_CARDS
		self.PDEALS_DONE = 1 + self.NUM_PDEALS              # 1 nullevent + 2 PDEALs ∀ player
		self.BLINDS_DONE = 1 + self.NUM_PDEALS + NUM_BLINDS # 1 nullevent + 2 PDEALs ∀ player + 2 blinds
		self.SB_STEP     = self.NUM_PDEALS + 1
		self.BB_STEP     = self.NUM_PDEALS + 2

	# Initializes from a history array and a bunch of individual game parameters
	cdef void      __MANUAL_INIT__( self, uint2 history, uint nPlayers, uint bpos, uint smallBlind, uint1 initStacks ): #noexcept:

		self.History          = history
		self.PLAYER_COUNT     = nPlayers
		self.ButtonPos        = bpos
		self.SmallBlindAmt    = smallBlind
		self.InitialStacks    = initStacks # [ 0,p1Stack,p2Stack ]
		self.SmallBlindPlayer = self.ButtonPos if self.PLAYER_COUNT==2 else NextNonDealer( self.ButtonPos )
		self.BigBlindPlayer   = NextNonDealer( self.SmallBlindPlayer )
		self.BigBlindAmt      = self.SmallBlindAmt*2 # TODO: GENERALIZE THIS
		self.hLen             = self.History.shape[ 0 ]
		
		# Useful early-game history indices; tells us first hist step AFTER certain stuff has been done
		self.NUM_PDEALS  = self.PLAYER_COUNT * MAX_HOLE_CARDS # = 2*2 = 4 for headsup
		self.PDEALS_DONE = 1 + self.NUM_PDEALS                # 1 nullevent + 2 PDEALs/player
		self.BLINDS_DONE = 1 + self.NUM_PDEALS + NUM_BLINDS   # 1 nullevent + 2 PDEALs/player + 2 blinds
		self.SB_STEP     = self.NUM_PDEALS + 1
		self.BB_STEP     = self.NUM_PDEALS + 2

	# Returns an array describing the game's initial conditions
	cdef uint1       InitialConditions( self ): #noexcept:

		cdef uint1 ic = pyarr( ARR_TMPLT_I, NUM_ICONDS, zero=False )
		ic[ NPLR ] = self.PLAYER_COUNT
		ic[ BPOS ] = self.ButtonPos
		ic[ SBAM ] = self.SmallBlindAmt
		ic[ STK: ] = self.InitialStacks # [ 0,p1Stack,p2Stack ]
		return ic

	# Returns last game event from history; options to start from a certain point or filter by event type
	cdef gameevent   LastEvent( self, uint from_point=0, uint of_type=NULLEVENT ): #noexcept:

		if from_point >= self.hLen: 
			return gameevent()

		# We take this to mean that the caller is looking for literally the last event to happen
		if of_type==NULLEVENT:      
			return gameevent( from_array=self.History[ self.hLen-1 ] )

		# Iterate backwards from latest event until a matching event is found then return it
		cdef uint s
		for s from self.hLen > s >= from_point:
			if self.History[ s,TYPE ]==of_type: 
				return gameevent( from_array=self.History[ s ] )

		# Return NULLEVENT if above loop finishes without finding a matching event
		return gameevent() 

	# Returns array of hole card vecs for specified player; Fill_To_Max inserts 0s for undealt cards
	cdef uint2       HoleCards( self, uint for_player, bint Fill_To_Max=FALSE ): #noexcept:

		cdef:
			uint  stop   = self.PDEALS_DONE if self.hLen > self.PDEALS_DONE else self.hLen, c=0, s
			uint2 hCards = cyarr( (MAX_HOLE_CARDS,CVEC_SIZE), UINTSIZE, 'I' )
			uint1 e

		# Scan over history, look for deal events to specified player, collect dealt cards
		for s from 0 <= s < stop:
			e = self.History[ s ]
			if e[ DEALTO ]==for_player:
				hCards[ c ] = CardOps.CardVector( e[ CDEALT ] )
				c+=1 

		if (Fill_To_Max) and (c!=MAX_BOARD_CARDS): 
			hCards[ c:,: ]=0

		return hCards if Fill_To_Max else hCards[ :c ]

	# Returns array of board card vectors; Fill_To_Max as above
	cdef uint2       BoardCards( self, bint Fill_To_Max=FALSE ): #noexcept:

		cdef:
			uint  c=0, s
			uint2 bCards = cyarr( (MAX_BOARD_CARDS,CVEC_SIZE), UINTSIZE, 'I' )
			uint1 e

		# Scan over history, look for board deal events, collect dealt cards
		for s from 0 <= s < self.hLen:
			e = self.History[ s ]
			if e[ TYPE ]==BOARDDEAL:
				bCards[ c ] = CardOps.CardVector( e[ CDEALT ] )
				c+=1 

		if (Fill_To_Max) and (c!=MAX_BOARD_CARDS): 
			bCards[ c:,: ]=0

		return bCards if Fill_To_Max else bCards[ :c ]

	# Returns array of all dealt cards; Validate arg is for eval ops where opponents are dealt "blank" cards
	cdef uint2       AllDealtCards( self, bint Validate=FALSE ): #noexcept:

		cdef:
			uint  c=0, s
			uint2 allCards = cyarr( (MAX_DEALT_CARDS, CVEC_SIZE), UINTSIZE, 'I' )
			uint1 e

		# Scan over history, find deal events, collect dealt cards
		for s from 0 <= s < self.hLen:
			e = self.History[ s ]

			if Is_Dealer_Action( e ):

				if not Validate:
					allCards[ c ] = CardOps.CardVector( e[ CDEALT ] )
					c+=1 

				elif self.History[ s,CDEALT ]!=0:
					allCards[ c ] = CardOps.CardVector( e[ CDEALT ] )
					c+=1 

		return allCards[ :c ]

	# Returns card vec array for all cards which haven't yet been dealt, Validate arg as above
	cdef uint2       AvailableDeck( self, bint Include_Gaps=FALSE, bint Validate=FALSE ): #noexcept:
		return CardOps.FilteredDeck( excludeCards=self.AllDealtCards( Validate ), Include_Gaps=Include_Gaps )

	# Counts num deal events in history, with filters for dealTo player & deals triggered by all-in actions
	cdef uint        NumDeals( self, uint to_player=ANY, bint Include_AllIn=TRUE ): #noexcept:

		cdef uint  nDeals=0, s
		cdef uint1 e

		for s from 1 <= s < self.hLen:
			e = self.History[ s ]

			if Is_Dealer_Action( e ):
				if (to_player==ANY) or (e[ DEALTO ]==to_player):
					if (not <bint>e[ IS_ALLIN ]) or (Include_AllIn):
						nDeals += 1

		return nDeals

	# Counts number of folds that have occurred
	cdef uint        NumFolds( self ): #noexcept:

		cdef uint nFolds=0, s
		for s from self.BLINDS_DONE <= s < self.hLen: 
			nFolds += <uint>(self.History[ s,TYPE ]==FOLD)

		return nFolds

	# Counts number of players that have gone all-in
	cdef uint        NumAllIns( self ): #noexcept:

		cdef uint  nAllIns=0, s
		cdef uint1 e

		for s from self.BLINDS_DONE <= s < self.hLen:
			e = self.History[ s ]
			nAllIns += <uint>( (e[ PLAYEDBY ] != DEALER) and (e[ IS_ALLIN ]) )

		return nAllIns

	# Calculates total each player has put into pot, option to start from a specific history step
	# TODO: Determine whether we need to correct ALLIN handling for >2pl
	cdef uint1       BetTotals( self, uint from_point=1 ): #noexcept:

		cdef uint1 bTotals = pyarr( ARR_TMPLT_I, self.PLAYER_COUNT+1, zero=True ), e
		cdef uint  s, p
		
		for s from from_point <= s < self.hLen:
			e = self.History[ s ]
			p = e[ PLAYEDBY ]
			bTotals[ p ] += e[ BETTOTAL ]
			
		return bTotals

	# Gets array of player IDs for all players in the game, regardless of player status
	cdef uint1       AllPlayers( self ): #noexcept:

		cdef uint  p
		cdef uint1 allPlayers = pyarr( ARR_TMPLT_I, self.PLAYER_COUNT+1, zero=False )

		for p from 0 <= p <= self.PLAYER_COUNT: 
			allPlayers[ p ] = p

		return allPlayers

	# Returns array of all players who are still able to act (i.e. are not folded or all-in)
	# Get allplayers ⟶ ∀p, set allplayers[p]=0 if p is folded/all-in ⟶ collect all p where allplayers[p]≠0 
	cdef uint1       ActivePlayers( self ): #noexcept:

		cdef uint  activeNum=0, s, p
		cdef uint1 activeStatus = self.AllPlayers(), aPlayers = pyarr( ARR_TMPLT_I, self.PLAYER_COUNT, zero=False ), e

		for s from self.BLINDS_DONE <= s < self.hLen:
			e = self.History[ s ]
			p = e[ PLAYEDBY ]
			if (e[ TYPE ]==FOLD) or (e[ IS_ALLIN ]): 
				activeStatus[ p ] = 0

		for p from 1 <= p <= self.PLAYER_COUNT:
			if activeStatus[ p ] != 0:
				aPlayers[ activeNum ] = p
				activeNum+=1

		return aPlayers[ :activeNum ]

	# Sometimes useful to know how many players are active without needing to know who they are
	cdef uint        NumActivePlayers( self ): #noexcept:
		return <uint>(self.PLAYER_COUNT - self.NumFolds() - self.NumAllIns())

	# Returns history index where the current betting round started
	cdef uint        CurrentRoundStart( self ): #noexcept:

		# Round starts at most recent deal not triggered by an all-in, so iter backwards to find it
		cdef uint  s
		cdef uint1 e
		for s from self.hLen > s >= 0:
			e = self.History[ s ]
			if (e[ PLAYEDBY ]==DEALER) and not <bint>(e[ IS_ALLIN ]): 
				return s

		return 0

	# Returns arr of stack refunds resulting from all-in calls falling short of required call amount
	cdef uint1       StackAdjustments( self ): #noexcept:

		cdef uint1 adjustments = pyarr( ARR_TMPLT_I, self.PLAYER_COUNT+1, zero=True ), e
		cdef uint  P = self.BigBlindPlayer, prevP, s # P starts = BB since loop starts right after big blind

		# Scan over history, find all allindiffs
		for s from self.BLINDS_DONE <= s < self.hLen:
			e = self.History[ s ]
			
			if not Is_Dealer_Action( e ):
				prevP = P
				P     = e[ PLAYEDBY ]

				if (e[ IS_ALLIN ]) and (e[ TYPE ]==CALL): 
					adjustments[ prevP ] += e[ ALLINDIFF ]

		return adjustments

	# Returns array of current stacks for all players
	# Get bet totals ⟶ get adjustments ⟶ ∀ p, stacks[p] = startingStack - btotals[p] + adjustments[p]
	cdef uint1       CurrentStacks( self ): #noexcept:

		cdef uint  p
		cdef uint1 bTotals     = self.BetTotals(),                                                                     \
				   adjustments = self.StackAdjustments(),                                                              \
				   stacks      = pyarr( ARR_TMPLT_I, self.PLAYER_COUNT+1, zero=False )
		stacks[ 0 ] = 0
		
		for p from 1 <= p <= self.PLAYER_COUNT: 
			stacks[ p ] = self.InitialStacks[ p ] - bTotals[ p ] + adjustments[ p ]

		return stacks

	# Gets current round ID: PREFLOP = 0, FLOP = 1, TURN = 2, RIVER = 3
	cdef uint        CurrentStreet( self ): #noexcept:

		# Only board deals not triggered by all-ins advance round num
		cdef uint nbDeals = self.NumDeals( to_player=BOARD, Include_AllIn=FALSE ), s
		if nbDeals==0:        return PREFLOP
		if 1 <= nbDeals <= 3: return FLOP
		if nbDeals==4:        return TURN
		if nbDeals==5:        return RIVER

	# Is current acting player posting a blind? 0 = not posting any blind, 1 = posting SB, 2 = posting BB
	cdef inline uint BlindState( self ): #noexcept:

		# NullEvent + 2 PDEALs ∀ player ⇒ currently posting SB
		if self.hLen == 1 + self.NUM_PDEALS: 
			return SB 

		# NullEvent + 2 PDEALs ∀ player + 1 SB ⇒ currently posting BB
		elif self.hLen == 2 + self.NUM_PDEALS: 
			return BB 

		return 0

	# Checks whether current hand's deals and blinds have occured yet, useful for short circuits.
	cdef inline bint Betting_Has_Started( self ): #noexcept:
		return self.hLen > self.BLINDS_DONE

	# Returns sequence of AllIn event & its direct successor, else NULLEVENT for either not present
	# Very helpful for executing game logic around all-ins
	cdef uint2       AllInSequence( self ): #noexcept:

		cdef uint  s
		cdef uint2 eSeq = cyarr( (2,EVEC_SIZE), UINTSIZE, 'I' )

		# Init event sequence with NULLEVENTS, only overwrite if relevant events have occurred
		eSeq[:] = 0
		for s from self.BLINDS_DONE <= s < self.hLen:

			if self.History[ s,IS_ALLIN ]: # found an all-in event
				eSeq[ 0 ] = self.History[ s ] 
				
				if s+1 < self.hLen: # response event has happened
					eSeq[ 1 ] = self.History[ s+1 ] 

				break

		return eSeq

	# Has an allin triggered a showdown? 
	# Check exclusion conditions ⟶ check for AllIn seq ⟶ check if all bcards dealt yet
	cdef bint        Awaiting_AllIn_Deal( self ): #noexcept:

		if not self.Betting_Has_Started():
			return FALSE

		# On river ⇒ all cards already dealt ⇒ no more deals needed
		if self.CurrentStreet()==RIVER:
			return FALSE 

		cdef uint2 allInSeq     = self.AllInSequence()
		cdef uint  nBCards      = self.NumDeals( BOARD ),                                                              \
				   allInType    = allInSeq[ 0,TYPE ],                                                                  \
				   responseType = allInSeq[ 1,TYPE ]

		# If no all-ins have happened, nothing to handle
		if allInType==NULLEVENT:
			return FALSE 

		# AllIn calls immediately trigger deal sequence
		if allInType==CALL:
			return nBCards < MAX_BOARD_CARDS 

		if allInType==RAISE:
			return FALSE if (responseType==NULLEVENT or responseType==FOLD) else nBCards < MAX_BOARD_CARDS 
		# 				    ￪ waiting for response ￪    ￪  game is over  ￪      ￪ need to deal more cards ￪

	# Are we awaiting a player (i.e. nondealer) fold/call response to an all-in action?
	cdef bint        Awaiting_AllIn_Response( self ): #noexcept:

		if not self.Betting_Has_Started(): 
			return FALSE

		cdef uint2 allInSeq  = self.AllInSequence()
		cdef uint  allInType = allInSeq[ 0,TYPE ], responseType = allInSeq[ 1,TYPE ]

		return TRUE if ((allInType==RAISE) and (responseType==NULLEVENT)) else FALSE

	# Does the dealer have more cards to deal before betting starts for this round?
	cdef bint        Round_Deals_Done( self ): #noexcept:

		cdef uint nbDeals = self.NumDeals( to_player=BOARD, Include_AllIn=FALSE )
		cdef bint Flop_Deals_Done  = nbDeals==3,                                                                       \
				  Turn_Deals_Done  = nbDeals==4,                                                                       \
				  River_Deals_Done = nbDeals==5

		# Preflop deals done if all players have gotten hcards
		if nbDeals==0:
			return self.hLen >= self.NUM_PDEALS+1 # +1 to account for initial NULLEVENT

		else:          
			return ( (Flop_Deals_Done) or (Turn_Deals_Done) or (River_Deals_Done) ) 

	# Checks whether all specified players have acted since from_point, useful game logic helper.
	cdef bint        Players_Have_Acted( self, uint1 players, uint from_point=0, bint Include_Blinds=FALSE ): #noexcept:

		if self.hLen - from_point < players.shape[ 0 ]: 
			return FALSE # Impossible for all p to have acted in this case

		cdef uint nP = players.shape[ 0 ], s, p, player
		cdef bint P_Has_Acted

		# ∀p ∈ players, scan from_point to hLen for p action, return FALSE if any player found who hasn't acted
		for p from 0 <= p < nP:
			player      = players[ p ]
			P_Has_Acted = FALSE

			for s from from_point <= s < self.hLen:

				if self.History[ s,PLAYEDBY ]==player: 

					if (Include_Blinds) or (s >= self.BLINDS_DONE): 
						P_Has_Acted = TRUE
						break

			if not P_Has_Acted: 
				return FALSE

		return TRUE # Loop never found a player who hasn't done something yet

	# Do all players' bet totals since from_point match? Part of round-ending logic.
	# ASSUMES p IS ACTIVE ∀p ∈ for_players. Checking bet equality for inactive p breaks game logic.
	cdef bint        All_Bets_Match( self, uint1 for_players, uint from_point=0 ): #noexcept:

		if for_players.shape[ 0 ] <= 1:
			return TRUE

		cdef uint1 bTotals = self.BetTotals( from_point )
		cdef uint  matchMe = bTotals[ for_players[0] ], nP = for_players.shape[ 0 ], p

		for p from 0 <= p < nP:
			if bTotals[ for_players[p] ] != matchMe: 
				return FALSE

		return TRUE # Never found a player with an unmatching bet total

	# Is the current betting round over? Things that end rounds:
	# 1) Things that end games, 2) All active players have acted & all their bets match
	# Only ever used for Is_Terminal and ActingPlayer; written mindful of checks that happen BEFORE it there
	# TODO: GENERALIZE ALLIN HANDLING HERE FOR >2pl BECAUSE IT DEFINITELY DOES NOT WORK FOR THAT CASE ATM
	cdef bint        Current_Round_Over( self ): #noexcept:

		if not self.Betting_Has_Started():
			return FALSE # duh

		if self.NumFolds()==self.PLAYER_COUNT-1:
			return TRUE  # If last man standing, game is over, so round is over...duh

		if not self.Round_Deals_Done():
			return FALSE # ...duh ಠ_ಠ

		cdef uint  rndStart
		cdef uint1 aPlayers

		# In heads-up, only time round not over with >0 allins is if awaiting allin response (RIGHT???)
		if self.NumAllIns() > 0:
			return not self.Awaiting_AllIn_Response()

		# Given above checks, we don't have to consider any special AllIn or Preflop conditions
		else: 
			aPlayers = self.ActivePlayers()
			rndStart = self.CurrentRoundStart()
			return self.Players_Have_Acted( aPlayers,rndStart ) and self.All_Bets_Match( aPlayers,rndStart )
	
	# Checks whether the current node is terminal (i.e. game-ending). Things that end games:
	# 1) <=1 actives (& not awaiting allin deal), 2) round==RIVER & round is over
	cdef bint        Is_Terminal( self ): #noexcept:

		cdef uint1 lastAction = self.History[ self.hLen-1 ]
		
		if not self.Betting_Has_Started(): 
			return FALSE

		if self.Awaiting_AllIn_Deal():     
			return FALSE

		if self.Awaiting_AllIn_Response(): 
			return FALSE

		if lastAction[ TYPE ]==RAISE:      
			return FALSE

		if self.NumActivePlayers() <= 1:   
			return TRUE # Can only safely do this since we've done Awaiting_AllIn_* checks

		# Only remaining terminal case now is non-allin showdown, i.e. CurrentStreet==RIVER & Current_Round_Over
		cdef bint Round_Over = self.Current_Round_Over(), Showdown = Round_Over and self.CurrentStreet()==RIVER

		if not Round_Over: 
			return FALSE

		elif Showdown:     
			return TRUE

		else:              
			return FALSE

	# Returns a preceding node at a specified history point (immediate predecessor if unspecified)
	cdef gamenode    Predecessor( self, uint with_hlen=0 ): #noexcept:

		cdef:
			uint  hlen    = with_hlen if with_hlen > 0 else self.hLen-1
			uint1 iConds  = self.InitialConditions()
			uint2 subHist = self.History.copy()[ :hlen ]
			
		return gamenode( history=subHist, initialConditions=iConds )

	# Is the current node a dealer node?
	cdef bint        Is_Dealer_Position( self ): #noexcept:
		return ( (not self.Round_Deals_Done()) or (self.Awaiting_AllIn_Deal()) or (self.Current_Round_Over()) )

	# Is this the first step in the current round? 
	cdef bint        At_Round_Start( self ): #noexcept:
		return self.hLen-1==self.CurrentRoundStart()

	# Gets player ID expected to act now (or at some prev point). ActingPlayer ≠ prevPlayer+1 when:
	# 1) Awaiting deal, 2) posting blinds, 3) at round start
	cdef uint        ActingPlayer( self, uint at_point=0 ): #noexcept:

		if at_point > 0: 
			return self.Predecessor( with_hlen=at_point+1 ).ActingPlayer()

		cdef uint lastPlayer = self.History[ self.hLen-1,PLAYEDBY ], blindState = self.BlindState()

		if self.Is_Dealer_Position(): 
			return DEALER

		if blindState==SB:            
			return self.SmallBlindPlayer

		if blindState==BB:            
			return self.BigBlindPlayer

		if self.At_Round_Start():     
			return self.BigBlindPlayer

		return NextNonDealer( lastPlayer )

	# Special case of Predecessor, allows excluding dealer nodes; returns copy of self if empty hist.
	cdef gamenode    ParentNode( self, bint Include_Deals=FALSE ): #noexcept:

		if self.hLen==1:
			return gamenode( history=self.History.copy(), initialConditions=self.InitialConditions() )

		if Include_Deals:
			return self.Predecessor()

		# Ok, definitely want to exclude dealer nodes at this point
		cdef:
			uint1    iConds = self.InitialConditions()
			uint2    subHistory
			gamenode stepNode
			uint     subLen

		# Scan history backwards, return first nondealer node found
		for subLen from self.hLen-1 >= subLen >= 1:
			subHistory = self.History[ :subLen ]
			stepNode   = gamenode( history=subHistory, initialConditions=iConds ) 
			
			if subLen==1 or stepNode.ActingPlayer() != DEALER: 
				return stepNode

	# Returns history array of only the current round.
	cdef uint2       CurrentRoundHist( self ): #noexcept:
		return self.History[ self.CurrentRoundStart(): ]

	# Calculates & returns current pot size, accounting for allin adjustments.
	cdef uint        PotSize( self ): #noexcept:

		cdef uint  pot=0, s
		cdef uint1 e

		for s from self.PDEALS_DONE <= s < self.hLen:
			e    = self.History[ s ]
			pot += (e[ BETTOTAL ] - e[ ALLINDIFF ])

		return pot

	# Calls CardOps hand evaluator to score all players' hands; hScores[0] = best hand's score.
	cdef uint1       HandScores( self ): #noexcept:

		cdef:
			uint  bestScore = MAX_HIGH_CARD, p
			uint1 hScores   = pyarr( ARR_TMPLT_I, self.PLAYER_COUNT+1, zero=False )
			uint2 bCards    = self.BoardCards()

		for p from 1 <= p <= self.PLAYER_COUNT:
			hScores[ p ] = CardOps.HandEval( self.HoleCards( p ), bCards )
			
			if hScores[ p ] < bestScore: # Yeah, lower scores are actually better hands
				bestScore = hScores[ p ]

		hScores[ 0 ] = bestScore
		return hScores

	# Determines winning player(s) from hand scores (can be multiple if tie)
	cdef uint1       Winners( self, uint1 from_scores ): #noexcept:

		cdef:
			uint1 winners   = pyarr( ARR_TMPLT_I, self.PLAYER_COUNT, zero=False )
			int   bestScore = from_scores[ 0 ] # best hand score always lives here
			uint  numWinners=0, p

		for p from 1 <= p <= self.PLAYER_COUNT:
			if from_scores[ p ]==bestScore: 
				winners[ numWinners ] = p
				numWinners += 1

		return winners[ :numWinners ]

	# Returns array of showdown net profits for all players, handles ties and all-in adjustments
	# Get hScores ⟶ find winner(s) ⟶ distribute pot across winners ⟶ subtract p's bets from p's payout ∀ p
	cdef int1        ShowdownResults( self, uint potSize, uint1 betTotals, uint1 stackAdjustments ): #noexcept:

		cdef:
			int1  results   = pyarr( ARR_TMPLT_i, self.PLAYER_COUNT+1, zero=False )
			uint1 hScores   = self.HandScores(),  winners  = self.Winners( from_scores=hScores )
			uint  bestScore = <uint>hScores[ 0 ], nWinners = winners.shape[ 0 ], divPot = potSize//nWinners, p

		results[ 0 ] = 0
		for p from 1 <= p <= self.PLAYER_COUNT:
			results[ p ]  = divPot - betTotals[ p ] if hScores[ p ]==bestScore else 0 - betTotals[ p ]
			results[ p ] += stackAdjustments[ p ] # account for all-in adjustments if present

			if (hScores[ p ]==bestScore) and (nWinners==1): 
				results[ 0 ] = p # setting results[0] = winnerID enables a nice short-circuit later

		return results

	# Executes game-scoring logic, returns net profit for all players:
	# Short-circuit if we don't need to score hands, else distribute profits according to hScores
	# TODO: Generalize to >2pl
	cdef int1        GameResults( self ): #noexcept:

		cdef:
			uint1 bTotals  = self.BetTotals(),                                                                         \
				  aPlayers = self.ActivePlayers(),                                                                     \
				  endgame  = self.History[ self.hLen-1 ],                                                              \
				  stackAdj = self.StackAdjustments()
			uint2 allInSeq = self.AllInSequence() 
			uint  nActive  = aPlayers.shape[ 0 ], pot = self.PotSize(), unadjustedNet, winner, p
			int1  results  = pyarr( ARR_TMPLT_i, self.PLAYER_COUNT+1, zero=False )

		# In heads-up, FOLD endgames imply two possible cases (neither requires hand scoring):
		# 1) nActive==1: Winner is last man standing. 
		# 2) nActive==0: Endgame seq must be ALLIN ⟶ FOLD. Winner is the all-in player.
		if endgame[ TYPE ]==FOLD:
			winner = aPlayers[ 0 ] if nActive > 0 else allInSeq[ 0,PLAYEDBY ]

			results[ 0 ] = winner			
			for p from 1 <= p <= self.PLAYER_COUNT:
				unadjustedNet = pot - bTotals[ p ] if p==winner else 0 - bTotals[ p ]
				results[ p ]  = unadjustedNet + stackAdj[ p ]

			return results

		# Endgame not FOLD ⇒ endgame must be showdown
		else: 
			return self.ShowdownResults( pot, bTotals, stackAdj )

	# Finds best hand the specified player can make, given dealt cards. Returns array of card vecs.
	cdef uint2       CurrentBestHand( self, uint for_player ): #noexcept:

		cdef uint2 holeCards = self.HoleCards( for_player ),                                                           \
				   bestFive  = CardOps.BestHand( holeCards, self.BoardCards() )
		cdef str   rankClass = CardOps.RankClass( self.HandScores()[ for_player ] ) 
		
		return CardOps.find_winning_cards( bestFive, holeCards, rankClass )

	# Returns ID of most recent player to get ALL of their hole cards
	cdef inline uint LastCompletedPDeal( self ): #noexcept:
		return (self.hLen-1) // MAX_HOLE_CARDS

	# Returns ID for who the next deal should go to
	cdef uint        DealTo( self ): #noexcept: 

		# After pdeals are done, all remaining deals go to board
		if self.hLen >= self.PDEALS_DONE: 
			return BOARD 

		# If pdeals not done, just return the next player who hasn't gotten all their cards	
		else:
			return self.LastCompletedPDeal()+1

	# Called at dealer nodes, generates contextually appropriate deal event:
	# Get current deck ⟶ decide deal target from round & AllIn info ⟶ draw card from deck ⟶ deal to target
	cdef gameevent   Deal( self ): #noexcept:

		cdef:
			bint  AllIn_Deal = self.Awaiting_AllIn_Deal()
			uint  dealTo     = self.DealTo(), dealType = BOARDDEAL if dealTo==BOARD else PLAYERDEAL
			uint1 dealCard   = CardOps.Draw( from_deck=self.AvailableDeck() )

		return gameevent( eventType=dealType, Is_AllIn=AllIn_Deal, dealTo=dealTo, cDealt=dealCard[ CARD ] )

	# Just a nice interface for getting some subset of the current node's history.
	cdef uint2       SubHistory( self, uint of_length ): #noexcept:
		return self.History.copy()[ :of_length ]

	# Orchestrates gamestate evolution: appends new event to current history, returns resulting node.
	cdef gamenode    Successor( self, gameevent e ): #noexcept:

		cdef uint2 newHist    = cyarr( (self.hLen+1, EVEC_SIZE), UINTSIZE, 'I' )
		newHist[ :self.hLen ] = self.History.copy()
		newHist[ self.hLen ]  = e.to_array()
		return gamenode( history=newHist, initialConditions=self.InitialConditions() )

	# Counts number of times specified player (defaults to all non-dealer) has had to act .
	cdef uint        NumDecisionPoints( self, uint for_player=ANY ): #noexcept:

		cdef uint nDecisions=0, s

		# Case for counting total decision points by either non-dealer player
		if for_player==ANY:
			for s from self.BLINDS_DONE <= s < self.hLen: 
				nDecisions += <uint>(self.History[ s,PLAYEDBY ]!=DEALER)

		else:
			for s from self.BLINDS_DONE <= s < self.hLen: 
				nDecisions += <uint>(self.History[ s,PLAYEDBY ]==for_player)

		return nDecisions

	# DEPRECATED: Generates a unique node identifier. Don't use this, it's extremely slow.
	cdef ll          GTKey_old( self ): #noexcept:
		return hash( str(NP( self.History,dtype=uintc ).tobytes()) )

	# Much faster manual hashing of history array to generate a unique gamenode ID
	cdef ll          GTKey( self ): #noexcept:

		cdef:
			uint  rows = self.History.shape[ 0 ], cols = self.History.shape[ 1 ], i, j, val
			ll    gtKey = <ll>14695981039346656037ULL
			uchar byte

		for i from 0 <= i < rows:
			for j from 0 <= j < cols:
				val    = self.History[ i,j ]

				byte   = <uchar>(val & 0xFF) 
				gtKey ^= byte
				gtKey *= 1099511628211LL

				byte   = <uchar>((val >> 8) & 0xFF) 
				gtKey ^= byte
				gtKey *= 1099511628211LL

				byte   = <uchar>((val >> 16) & 0xFF) 
				gtKey ^= byte
				gtKey *= 1099511628211LL

				byte   = <uchar>((val >> 24) & 0xFF)
				gtKey ^= byte
				gtKey *= 1099511628211LL

		return gtKey

	# Just dumps a bunch of info about the current node for human-readable gamestate inspection.
	cdef void        summary( self, bint Compact=0 ): #noexcept:

		cdef:

			uint p, step, d, c, nD = self.NumDeals(), rStart = self.CurrentRoundStart(), nP = self.PLAYER_COUNT,       \
			     bpos  = self.ButtonPos, sbp = self.SmallBlindPlayer, bbp = self.BigBlindPlayer,                       \
			     sbamt = self.SmallBlindAmt, bbamt = self.BigBlindAmt

			uint2 bCards    = self.BoardCards(),                                                                       \
				  hCardsP1  = self.HoleCards( for_player=1 ),                                                          \
				  hCardsP2  = self.HoleCards( for_player=2 ),                                                          \
				  deck      = self.AvailableDeck(),                                                                    \
				  printDeck = self.AvailableDeck( Include_Gaps=TRUE )
			
			uint1 istacks    = self.InitialStacks[1:],                                                                 \
				  betTotals  = self.BetTotals(),                                                                       \
				  rBetTotals = self.BetTotals( from_point=rStart ),                                                    \
				  aPlayers   = self.ActivePlayers(),                                                                   \
				  stacks     = self.CurrentStacks()
			uint  dSize      = deck.shape[0]
			
			str tab       = '\t'*(not Compact),                                                                        \
				bPretty   = ''.join(  CardOps.PrettyCardStrings( bCards,    Compact=TRUE, Center=FALSE ) ),            \
				hPrettyP1 = ''.join(  CardOps.PrettyCardStrings( hCardsP1,  Compact=TRUE, Center=FALSE ) ),            \
				hPrettyP2 = ''.join(  CardOps.PrettyCardStrings( hCardsP2,  Compact=TRUE, Center=FALSE ) ),            \
				dPretty   = '|'.join( CardOps.PrettyCardStrings( printDeck, Compact=TRUE, Center=FALSE ) ),            \
				stepStr, cond
			
			list YN      = [ "NO", "YES" ],                                                                            \
				 hPretty = [ hPrettyP1, hPrettyP2 ],                                                                   \
				 streets = [ "PREFLOP", "FLOP", "TURN", "RIVER" ],                                                     \
				 players = [ "DEALER" ]+[ f"P{p}" for p in self.AllPlayers()[1:] ],                                    \
				 iConds  =                                                                                             \
				 	[ str( nP ), str( bpos ), str( sbp ), str( bbp ), str( sbamt ), str( bbamt ), str( NP(istacks) ) ]

			str  aPlayer = players[ self.ActingPlayer() ]

		if not Compact:
			print( '\n'+('='*100 ))
			print( "GAMENODE SUMMARY".center(100) )
			print( ('='*100)+'\n' )

			print( '\t' + ('='*60) )
			print( '\t' + f"INITIAL CONDITIONS".center(60) )
			print( '\t' + ('='*60) )
			print( '\t' + f"  nP  | | BPOS | |SBPLYR| |BBPLYR| |SB AMT| |BB AMT| |ISTACKS " )
			print( '\t' + "| |".join( [cond.rjust(6) for cond in iConds] ) )
			print()

			print( '\t' + ('='*50) )
			print( '\t' + f"STATE INFO".center(50) )
			print( '\t' + ('='*50) )
			print( '\t' + f"Node hLen".rjust(20)        + f" || {self.hLen}" )
			print( '\t' + f"Total Num Deals".rjust(20)  + f" || {nD}" )
			print( '\t' + f"Acting Player".rjust(20)    + f" || {aPlayer}" )
			print( '\t' + f"Active Players".rjust(20)   + f" || {list( aPlayers )}" )
			print( '\t' + f"Betting Started".rjust(20)  + f" || {YN[ self.Betting_Has_Started() ]}" )
			print( '\t' + f"Current Pot Size".rjust(20) + f" || {self.PotSize()}" )
			print( '\t' + f"Game Bet Totals".rjust(20)  + f" || {list( betTotals )}" )
			print( '\t' + f"Current Stacks".rjust(20)   + f" || {list( stacks )}" )
			print( '\t' + f"Is_Terminal".rjust(20)      + f" || {YN[ self.Is_Terminal() ]}" )
			if self.Is_Terminal(): 
				print( '\t' + f"Hand Scores".rjust(20)  + f" || {list( self.HandScores() )}" )
			print()

			print( '\t' + ('='*50) )
			print( '\t' + f"CARD INFO".center(50) )
			print( '\t' + ('='*50) )
			print( '\t' + f"Deck ({dSize})".rjust(10)     + f" || |{dPretty}|" )
			print( '\t' + f"B Cards".rjust(10)  + f" || {bPretty}" )
			print( '\t' + f"P1 Cards".rjust(10) + f" || {hPrettyP1}" )
			print( '\t' + f"P2 Cards".rjust(10) + f" || {hPrettyP2}" )
			print()

			print( '\t' + ('='*50) )
			print( '\t' + f"ROUND INFO".center(50) )
			print( '\t' + ('='*50) )
			print( '\t' + f"Current Street".rjust(25)            + f" || {streets[ self.CurrentStreet() ]}" )
			print( '\t' + f"Current round is over".rjust(25)     + f" || {YN[ self.Current_Round_Over() ]}" )
			print( '\t' + f"Current round start".rjust(25)       + f" || {rStart}" )
			print( '\t' + f"Round bet totals".rjust(25)          + f" || {list( rBetTotals )}" )
			print( '\t' + f"All P have acted in round".rjust(25) + f" || {YN[ self.Players_Have_Acted( aPlayers,rStart ) ]}" )
			print( '\t' + f"Last action was raise".rjust(25)     + f" || {YN[ self.LastEvent().Type==RAISE ]}" )
			print( '\t' + f"Awaiting allin deal".rjust(25)       + f" || {YN[ self.Awaiting_AllIn_Deal() ]}" )
			print( '\t' + f"Awaiting allin response".rjust(25)   + f" || {YN[ self.Awaiting_AllIn_Response() ]}" )
			print()

			print( f"\tGTKey     = {self.GTKey()}" )
			print( f"\tLastEvent = {self.LastEvent().ShortString()}" )

		if Compact: print()
		print( tab + ('='*85) )
		print( tab + f"GAME HISTORY".center(85) )
		print( tab + ('='*85) )
		print( tab + f" STEP | | TYPE | |PLAYER| | RAMT | | BTOT | |ALL_IN| |ALLIND| |DEALTO| |CDEALT " )
		print( tab + ('—'*85) )

		for step from 0 <= step <= self.hLen:
			if step < self.hLen:
				print( tab + f"{ gameevent( from_array=self.History[step] ).ShorterString( step ) }" )
			elif not self.Is_Terminal():
				print( tab + f"{str(step).rjust(6)}| |" + f"[[ CURRENT STEP - AWAITING {aPlayer} ACTION ]]".center(74) )

		cdef:
			list gameResults, winners = []
			uint winner, nWinners, i
			str  endStr, resStr

		if self.Is_Terminal():
			gameResults = list( self.GameResults() )
			nWinners    = gameResults[1:].count( max(gameResults[1:]) )
			resStr      = tab + f"Results: {gameResults}".center(85)

			if nWinners == 1: 
				winner = gameResults[0]
				endStr = tab + f"GAMEOVER, P{winner} WINS".center(85)

			else:

				for p in self.AllPlayers()[1:]:
					if gameResults[ p ] == max( gameResults[1:] ): 
						winners.append( p )

				endStr = tab + f"GAMEOVER, TIE BETWEEN: {winners}".center(85)

			print( tab+('='*85) )
			print( endStr )
			print( resStr )
			print( tab+('='*85) )

		if not Compact: print( ('='*100)+'\n' )
		else:           print( ('='*85)+'\n' )

	# More exhaustive than .summary(), prints all fields and outputs of most functions, use for deep debugging.
	cdef void        DIAGNOSTIC( self ): #noexcept:

		cdef:
			uint  rndStart    = self.CurrentRoundStart(), d
			uint1 aPlayers    = self.ActivePlayers()
			uint2 gapDeck     = self.AvailableDeck( Include_Gaps=TRUE ), allCards = self.AllDealtCards(),              \
				  bCards      = self.BoardCards(), h1Cards = self.HoleCards( 1 ), h2Cards = self.HoleCards( 2 ),       \
				  allInSeq    = self.AllInSequence()
			str   printDeck   = '|'.join( CardOps.PrettyCardStrings( gapDeck, Compact=TRUE, Center=FALSE ) ),          \
				  printBoard  =  ''.join( CardOps.PrettyCardStrings( bCards,  Compact=TRUE, Center=FALSE ) ),          \
				  printH1     =  ''.join( CardOps.PrettyCardStrings( h1Cards, Compact=TRUE, Center=FALSE ) ),          \
				  printH2     =  ''.join( CardOps.PrettyCardStrings( h2Cards, Compact=TRUE, Center=FALSE ) ),          \
				  printCards  =  ''.join( CardOps.PrettyCardStrings( allCards,Compact=TRUE, Center=FALSE ) ),          \
				  winnerStr
			tuple BLINDSTATES = ( "NOPE", "SB", "BB" )

		print( '\n'+('='*100) )
		print( ">>> GAMENODE DIAGNOSTIC <<<".center(100) )
		print( ('='*100)+'\n' )

		print( f"PLAYER_COUNT = ......{self.PLAYER_COUNT}" )
		print( f"NUM_PDEALS = ........{self.NUM_PDEALS}" )
		print( f"PDEALS_DONE = .......{self.PDEALS_DONE}" )
		print( f"BLINDS_DONE = .......{self.BLINDS_DONE}" )
		print( f"SB_STEP = ...........{self.SB_STEP}" )
		print( f"BB_STEP = ...........{self.BB_STEP}" )
		print( f"ButtonPosition = ....{self.ButtonPos}" )
		print( f"SmallBlindPlayer = ..{self.SmallBlindPlayer}" )
		print( f"BigBlindPlayer = ....{self.BigBlindPlayer}" )
		print( f"SmallBlindAmount = ..{self.SmallBlindAmt}" )
		print( f"BigBlindAmount = ....{self.BigBlindAmt}" )
		print( f"InitialStacks = .....{NP( self.InitialStacks )}" )
		print( f"hLen = ..............{self.hLen}" )
		print()

		print( f"InitialConditions() = ...........{list( self.InitialConditions() )}" )
		print( f"LastEvent() = ...................{self.LastEvent().ShortString()}" )
		print( f"HoleCards( 1 ) = ................{printH1}" )
		print( f"HoleCards( 2 ) = ................{printH2}" )
		print( f"BoardCards() = ..................{printBoard}" )
		print( f"AllDealtCards() = ...............{printCards}" )
		print( f"AvailableDeck() = ...............{printDeck}" )
		print( f"NumDeals( BOARD ) = .............{self.NumDeals( BOARD )}" )
		print( f"NumDeals( BOARD,allin=FALSE ) =  {self.NumDeals( to_player=BOARD, Include_AllIn=FALSE )}" )
		print( f"NumDeals( 1 ) = .................{self.NumDeals( 1 )}" )
		print( f"NumDeals( 2 ) = .................{self.NumDeals( 2 )}" )
		print( f"NumFolds() = ....................{self.NumFolds()}" )
		print( f"NumAllIns() = ...................{self.NumAllIns()}" )
		print( f"BetTotals() = ...................{list( self.BetTotals() )}" )
		print( f"BetTotals( rndStart ) = .........{list( self.BetTotals( from_point=rndStart ) )}" )
		print( f"AllPlayers() = ..................{list( self.AllPlayers() )}" )
		print( f"ActivePlayers() = ...............{list( self.ActivePlayers() )}" )
		print( f"NumActivePlayers() = ............{self.NumActivePlayers()}" )
		print( f"CurrentRoundStart() = ...........{self.CurrentRoundStart()}" )
		print( f"StackAdjustments() = ............{list( self.StackAdjustments() )}" )
		print( f"CurrentStacks() = ...............{list( self.CurrentStacks() )}" )
		print( f"CurrentStreet() = ...............{ROUNDNAMES[ self.CurrentStreet() ]}" )
		print( f"BlindState() = ..................{BLINDSTATES[ self.BlindState() ]}" )
		print( f"Betting_Has_Started() = .........{self.Betting_Has_Started()}" )
		print( f"AllInSequence() = " )
		print( f"\t{gameevent( from_array=allInSeq[ 0 ] ).ShortString()}" )
		print( f"\t{gameevent( from_array=allInSeq[ 1 ] ).ShortString()}" )
		print( f"Awaiting_AllIn_Deal() = .........{self.Awaiting_AllIn_Deal()}" )
		print( f"Awaiting_AllIn_Response() = .....{self.Awaiting_AllIn_Response()}" )
		print( f"Round_Deals_Done() = ............{self.Round_Deals_Done()}" )
		if aPlayers.shape[ 0 ] > 0: 
			print( f"Players_Have_Acted( rStart ) = ..{self.Players_Have_Acted( aPlayers, rndStart )}" )
			print( f"All_Bets_Match( rStart ) = ......{self.All_Bets_Match( aPlayers, rndStart )}" )
		print( f"Current_Round_Over() = ..........{self.Current_Round_Over()}" )
		print( f"Is_Terminal() = .................{self.Is_Terminal()}" )
		print( f"Is_Dealer_Position() = ..........{self.Is_Dealer_Position()}" )
		print( f"At_Round_Start() = ..............{self.At_Round_Start()}" )
		print( f"ActingPlayer() = ................{PLAYERNAMES[ self.ActingPlayer() ]} " )
		print( f"PotSize() = .....................{self.PotSize()}" )
		print( f"NumDecisionPoints() = ...........{self.NumDecisionPoints()}" )
		print( f"GTKey() = .......................{self.GTKey()}" )

		if self.Is_Terminal():
			winnerStr = " (hands not scored, this is bollocks)" if (NP(self.HandScores())==0).all() else ''
			print( f"HandScores() = ..................{list( self.HandScores() )}" )
			print( f"Winners( hScores ) = ............{list( self.Winners( self.HandScores() ) )}" + winnerStr )
			print( f"ShowdownResults() = .............{list( self.ShowdownResults( self.PotSize(),self.BetTotals(),self.StackAdjustments() ))}" )
			print( f"GameResults() = .................{list( self.GameResults() )}" )

		# input( "\nYE?" )

		
# Utility function that returns a start-game node with empty history & specified game parameters
cdef gamenode RootNode( uint1 initialConditions=None, uint players=2, uint buttonPos=2, uint smallBlind=1, uint1 initStacks=None ): #noexcept:

	cdef uint2 nullHist = cyarr( (1,EVEC_SIZE), UINTSIZE, 'I' )
	nullHist[ 0 ] = gameevent().to_array()

	if initialConditions is None:
		return gamenode( nullHist, players, buttonPos, smallBlind, initStacks )
	else:
		return gamenode( history=nullHist, initialConditions=initialConditions )

# Gets a node with initial deal events done, useful for some testing/simulation stuff downstream
cdef gamenode DummyNode(): #noexcept
	
	cdef:
		uint1     dumbConds = cyarr( (NUM_ICONDS,), UINTSIZE, 'I' )
		gamenode  dumbNode
		gameevent dumbDeal
		uint      d=0

	dumbConds[:]=0
	dumbConds[ NPLR ] = NUM_PLAYERS
	dumbConds[ BPOS ] = 1   # ¯\_(ツ)_/¯
	dumbConds[ SBAM ] = 10  # ¯\_(ツ)_/¯
	dumbConds[ STK1 ] = 420 # ¯\_(ツ)_/¯
	dumbConds[ STK2 ] = 420 # ¯\_(ツ)_/¯

	dumbNode = RootNode( initialConditions=dumbConds )

	while d < dumbNode.NUM_PDEALS:
		dumbDeal = dumbNode.Deal()
		dumbNode = dumbNode.Successor( dumbDeal )
		d+=1

	return dumbNode
	

# *-* # 