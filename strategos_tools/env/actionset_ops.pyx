# distutils: language = c
# cython: language_level 3
# cython: profile = False


cimport cython
from cpython.array cimport clone as pyarr
from cython.view   cimport array as cyarr

from strategos_tools.env.event_ops cimport AllNonRaises, Is_Null

import numpy as np
from numpy import asarray as NP, uintc


# ==================================================================================================
# An actionset is exactly what it sounds like: A(I) is the set of actions available to I.POVplayer
# at infoset I. This class just takes an infoset I on init, goes through a bunch of poker logic to
# determine what actions are availble, then stores determined action boundary conditions in a way
# that allows us to generate available actions as we need them on the fly, instead of storing huge
# arrays of all actions themselves. Actions are generated/retrieved using an indexing system.
# ==================================================================================================


cdef class actionset:

	def __init__( self, infoset I ): 
		self.__INIT__( I )

	# Determines basic boundary conditions on available actions & calls helpers to fill out details.
	cdef void        __INIT__( self, infoset I ): #noexcept:
	
		self._I = I
		self.Player = I.POVplayer
		
		if not I.POV_Is_Acting_Player():
			self.MatchCall    = 0
			self.MinRaise     = 0
			self.MaxRaise     = 0
			self.NumNonRaises = 0
			self.NumRaises    = 0
			self.size         = 0

		else:
			self.CurrentStack  = I.CurrentStack()
			self.Posting_Blind = I.Is_Posting_Blind()
			self.MatchCall     = I.CallAmount()

			self.__find_available_nonraises()
			self.__find_available_raises()
			self.size = self.NumNonRaises + self.NumRaises

	# Builds explicit array of available non-raise actions (there aren't many of them, just store explicitly).
	cdef void        __find_available_nonraises( self ): #noexcept:

		# First define some constraining conditions on available actions
		cdef:
			bint      Pre_Flop       = self._I.CurrentStreet()==PREFLOP 
			uint      roundStart     = self._I.CurrentRoundStart() if (not Pre_Flop) else self._I._n.BLINDS_DONE 
			gameevent lastBet        = self._I.LastEvent( from_point=roundStart, of_type=RAISE )
			bint      Betting_Opened = lastBet.Type!=NULLEVENT,                                                        \
					  First_Action   = self._I.Is_First_Action()

		# Now, which nonraise actions are available to us?
		self.Can_Fold  = (self.Posting_Blind==FALSE) and ((Betting_Opened==TRUE) or (First_Action==TRUE))
		self.Can_Check = (self.Posting_Blind==FALSE) and (self.MatchCall==0)
		self.Can_Call  = (self.Posting_Blind==FALSE) and (self.Can_Check==FALSE) and (self.CurrentStack>0)

		# Determine specific properties of available nonraises
		cdef:
			bint  Call_In     = self.CurrentStack <= self.MatchCall # Would a call require us to go all-in?
			uint  callAmt     = self.MatchCall if (not Call_In) else self.CurrentStack,                                \
				  calldiff    = self.MatchCall - self.CurrentStack if Call_In else 0
			uint2 nonRaises   = AllNonRaises( self.Player, callAmt, Call_In, calldiff ),                               \
				  availableNR = cyarr( (NON_RAISE_ETYPES, EVEC_SIZE), UINTSIZE, 'I' )
			uint  n=0

		# Populate array of nonraises
		if self.Can_Fold:
			availableNR[ n ] = nonRaises[ FOLD ]
			n+=1
		if self.Can_Check:
			availableNR[ n ] = nonRaises[ CHECK ]
			n+=1
		if self.Can_Call:
			availableNR[ n ] = nonRaises[ CALL ]
			n+=1

		self.CheckCallIdx = <uint>self.Can_Fold # Idx of CHECK/CALL = 1 if fold is available, else 0
		self.NonRaises    = availableNR[ :n ]
		self.NumNonRaises = n

	# Uses diff between stack & min bet to find range of available raises.
	cdef void        __find_available_raises( self ): #noexcept:

		# Only FOLD/CALL allowed as responses to all-ins
		self.Can_Raise = (self.CurrentStack > self.MatchCall) and (not self._I.Awaiting_AllIn_Response())
		
		if self.Can_Raise:
			self.MinRaise  = self._I.MinRaise() # Minimum ALLOWED raise
			self.MaxRaise  = self._I.MaxRaise() # Maximum POSSIBLE raise
			
			# If max possible raise < min allowed raise, only available raise is all-in 
			self.NumRaises = 1 if self.MinRaise >= self.MaxRaise else (self.MaxRaise - self.MinRaise) + 1
			
		else: 
			self.NumRaises = 0

	# Gets action at aIdx: if nonraise, return stored array, else generate raise event using aIdx offset.
	cdef gameevent     at( self, uint aIdx ): #noexcept:
	
		# Already have nonraise event arrs stored, so just construct gameevent from one
		if aIdx < self.NumNonRaises:
			return gameevent( from_array=self.NonRaises[ aIdx ] )

		# Ok we need to generate a raise action, so first determine properties 
		cdef bint All_In = (self.NumRaises==1 or aIdx==self.size-1) and (not self.Posting_Blind)
		cdef uint rIdx   = aIdx - self.NumNonRaises,                                                                   \
				  rAmt   = self.MinRaise + rIdx if (not All_In) else self.CurrentStack - self.MatchCall,               \
				  bTot   = rAmt + self.MatchCall # if All_In this is just self.CurrentStack...duh

		return gameevent( eventType=RAISE, playedBy=self.Player, raiseAmt=rAmt, betTotal=bTot, Is_AllIn=All_In )

	# Helper for finding the Aset index of a given raise, just uses total bet offset from minraise.
	cdef inline uint __get_raise_aIdx( self, uint from_total_bet ): #noexcept:

		cdef uint raiseAmt = from_total_bet - self.MatchCall, raiseIdx = raiseAmt - self.MinRaise
		return self.NumNonRaises + raiseIdx

	# Useful util func downstream when we need the Aset index of some observed action.
	cdef inline uint   index_of( self, gameevent e ): #noexcept:

		if e.Type!=RAISE: 
			return 0 if e.Type==FOLD else self.CheckCallIdx

		if e.Type==RAISE: 
			return self.__get_raise_aIdx( from_total_bet=e.BetTotal )

	# Returns A as a matrix of event arrays. If A.size=0, AMat.shape=(0,EVEC_SIZE). God help you.
	cdef uint2         AMat( self ): #noexcept:
	
		cdef uint  mSize = self.size or 1, a
		cdef uint2 AMat  = cyarr( (mSize, EVEC_SIZE), UINTSIZE, 'I' )

		for a from 0 <= a < self.size: 
			AMat[ a ] = self.at( a ).to_array()

		return AMat[ :self.size ] # self.size==0 isn't actually a problem here, neat

	cdef gamenode      SourceNode( self ): #noexcept:
		return self._I._n

	# str like: "A = { NONRAISES: [ FOLD | CALL $CAMT ] || RAISERANGE: [ $RAMT($BTOT) | $RAMT($BTOT) ] }"
	cdef str           inline_summary( self ): #noexcept:

		cdef:
			list  nonRaises  = [] if self.NumNonRaises > 0 else [ 'NONE' ],                                            \
				  raiseRange = [] if self.Can_Raise else [ 'NONE' ]
			uint  a, eType
			uint1 nr
			str   nrStr, minrStr, maxrStr

		for a from 0 <= a < self.NumNonRaises:
			nr    = self.NonRaises[ a ]
			eType = nr[ TYPE ]
			nrStr = TYPENAMES[ eType ].strip() if eType!=CALL else TYPENAMES[ eType ] + f"${nr[ BETTOTAL ]}"
			nonRaises.append( nrStr )

		minrStr    = f"${self.MinRaise}(${self.MatchCall + self.MinRaise})"
		maxrStr    = f"${self.MaxRaise}(${self.MatchCall + self.MaxRaise})"
		raiseRange = [ maxrStr ] if self.NumRaises==1 else [ minrStr, maxrStr ]

		# Holy fucking string formatting
		return '{ NONRAISES: [ ' + ' | '.join( nonRaises ) + ' ] || RAISERANGE: [ ' + ' | '.join( raiseRange ) + ' ] }'

	# Just dumps a bunch of info for human-readable inspection.
	cdef void          summary( self ): #noexcept:

		cdef:
			uint1 e
			uint  minRAmt = self.MatchCall+self.MinRaise, maxRAmt = self.MatchCall+self.MaxRaise, P = self.Player
			bint  All_In  = not (<bint>self.Posting_Blind)

			gameevent minRaise =                                                                                       \
				gameevent( RAISE, playedBy=P, raiseAmt=self.MinRaise, betTotal=minRAmt )
			gameevent maxRaise =                                                                                       \
				gameevent( RAISE, playedBy=P, raiseAmt=self.MaxRaise, betTotal=maxRAmt, Is_AllIn=All_In )

		print( '\n\t'+('='*50 ) )
		print( '\t' + f"P{self.Player} ACTIONSET SUMMARY".center(50) )
		print( '\t'+('='*50) )	

		print( '\t' + f"Owner Stack".rjust(24)     + f" || {self.CurrentStack}" )
		print( '\t' + f"A.Posting_Blind".rjust(24) + f" || {bool(self.Posting_Blind)}" )
		print( '\t' + f"|A|".rjust(24)             + f" || {self.size}" )
		print( '\t' + f"A.Can_Fold".rjust(24)      + f" || {self.Can_Fold}" )
		print( '\t' + f"A.Can_Check".rjust(24)     + f" || {self.Can_Check}" )
		print( '\t' + f"A.Can_Call".rjust(24)      + f" || {self.Can_Call}" )
		print( '\t' + f"A.Can_Raise".rjust(24)     + f" || {self.Can_Raise}" )
		print( '\t' + f"A.MatchCall".rjust(24)     + f" || {self.MatchCall}" )
		print( '\t' + f"A.MinRaise".rjust(24)      + f" || {self.MinRaise}" )
		print( '\t' + f"A.MaxRaise".rjust(24)      + f" || {self.MaxRaise}" )
		print( '\t' + f"A.NumNonRaises".rjust(24)  + f" || {self.NumNonRaises}" )
		print( '\t' + f"A.NumRaises".rjust(24)     + f" || {self.NumRaises}" )
		print()

		print( '\t' + ('='*25) )
		print( '\t' + f"AVAILABLE NONRAISES".center(25) )
		print( '\t' + ('='*25) )
		if self.NumNonRaises > 0:
			for e in self.NonRaises: print( '\t' + f"{ gameevent( from_array=e ).ShortString() }" )
		else: print( '\t' + "[[ NO NONRAISES AVAILABLE ]]".center(25) )
		print()

		print( '\t' + ('='*25) )
		print( '\t' + f"RAISE RANGE".center(25) )
		print( '\t' + ('='*25) )
		if self.NumRaises > 0:
			if maxRaise.RaiseAmt > minRaise.RaiseAmt: print( '\t' + f"Min: { minRaise.ShortString() }" )
			print( '\t' + f"Max: { maxRaise.ShortString() }" )
		else: print( '\t' + f"[[ NO RAISES AVAILABLE ]]".center(25) )
		print()

	# More exhaustive than .summary(), prints all fields and outputs of most functions, use for deep debugging.
	cdef void          DIAGNOSTIC( self ): #noexcept:

		cdef uint  i 
		cdef uint1 e
		
		print( '\n'+('='*100 ))
		print( ">>> ACTIONSET DIAGNOSTIC <<<".center(100) )
		print( ('='*100)+'\n' )

		print( f"Player = .........P{self.Player}" )
		print( f"CurrentStack = ...{self.CurrentStack}" )
		print( f"MatchCall = ......{self.MatchCall}" )
		print( f"MinRaise = .......{self.MinRaise}" )
		print( f"MaxRaise = .......{self.MaxRaise}" )
		print( f"NumActions = .....{self.size}" )
		print( f"NumNonRaises = ...{self.NumNonRaises}" )
		print( f"NumRaises = ......{self.NumRaises}" )
		print( f"Posting_Blind = ..{['NOPE', 'SMALL', 'BIG'][ self.Posting_Blind ]}" )
		print( f"Can_Fold = .......{bool( self.Can_Fold )}" )
		print( f"Can_Check = ......{bool( self.Can_Check )}" )
		print( f"Can_Call = .......{bool( self.Can_Call )}" )
		print( f"Can_Raise = ......{bool( self.Can_Raise )}" )
		if self.NumNonRaises > 0:
			print( f"NonRaises = ..." )
			print( NP( self.NonRaises,dtype=uintc ) )
			for e in self.NonRaises: print( f"{ gameevent( from_array=e ).ShortString() }" )
		print()

		print( f"AMat() = ..." )
		for e in self.AMat(): print( '[' + ' '.join([ f"{i}".rjust(3) for i in NP( e,dtype=uintc ) ]) + ' ]' )
		print()

		cdef:
			object    RNG
			uint      randomIdx
			gameevent randomEvent
		if self.size > 0:
			RNG         = np.random.default_rng()
			randomIdx   = RNG.choice( self.size )
			randomEvent = self.at( randomIdx ) 

		print( f"A.at( randomIdx )         = {randomEvent.ShortString()}" )
		print( f"A.index_of( randomEvent ) = {self.index_of( randomEvent )} (should be == {randomIdx})" )

		input( "\nYE?" )


# *-* # 