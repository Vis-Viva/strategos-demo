# distutils: language = c
# cython: language_level 3
# cython: profile = False


from cpython.array cimport clone as pyarr
from cython.view   cimport array as cyarr

from strategos_tools.env.card_ops cimport CardVector, PrettyCardStrings

from numpy import asarray as NP, uintc


# ==================================================================================================
# A gameevent is just a a single action taken by a player. It is identified by:
# - Type:      FOLD, CHECK, CALL, RAISE, BOARDDEAL, PLAYERDEAL
# - PlayedBy:  DEALER (0), 1, 2
# - RaiseAmt:  By how much the player is raising the current stakes if type is RAISE
# - BetTotal:  Total player spend in this action, if a betting action
# - Is_AllIn:  Is player now all-in? If deal event, triggered by an all-in? Useful game logic flag.
# - AllInDiff: If all-in, by how much the bet fell short of min required bet. Useful game logic info.
# - DealTo:    If deal event, player who card is being dealt to
# - CDealt:    If deal event, cardID being dealt
# Basically just a size 8 uint array with some useful utility functions attached to it. Sequences of
# these form a game history.
# ==================================================================================================


cdef class gameevent:

	def __init__( self, uint eventType=NULLEVENT, uint playedBy=0, uint raiseAmt=0, uint betTotal=0, 
				  uint Is_AllIn=FALSE, uint allInDiff=0, uint dealTo=0, uint cDealt=0, 
				  uint1 from_array=None ):

		if from_array is None:
			self.__MANUAL_INIT__( eventType, playedBy, raiseAmt, betTotal, 
								  Is_AllIn, allInDiff, dealTo, cDealt )
		else:
			self.__AUTOINIT__( from_array )

	# Initializes from a specifically-structured predefined event array
	cdef void __AUTOINIT__( self, uint1 from_array ): #noexcept:

		self.Type      = from_array[ TYPE ]
		self.PlayedBy  = from_array[ PLAYEDBY ]
		self.RaiseAmt  = from_array[ RAISEAMT ]
		self.BetTotal  = from_array[ BETTOTAL ]
		self.Is_AllIn  = from_array[ IS_ALLIN ]
		self.AllInDiff = from_array[ ALLINDIFF ]
		self.DealTo    = from_array[ DEALTO ]
		self.CDealt    = from_array[ CDEALT ]

	# Initializes from manually-specified event parameters
	cdef void __MANUAL_INIT__( self, uint eventType, uint playedBy, uint raiseAmt, uint betTotal, 
							   uint Is_AllIn, uint allInDiff, uint dealTo, uint cDealt ): #noexcept:

		self.Type      = eventType
		self.PlayedBy  = playedBy
		self.RaiseAmt  = raiseAmt
		self.BetTotal  = betTotal
		self.Is_AllIn  = Is_AllIn
		self.AllInDiff = allInDiff
		self.DealTo    = dealTo
		self.CDealt    = cDealt

	cdef uint1 to_array( self ): #noexcept:

		cdef uint1 eventArr = pyarr( ARR_TMPLT_I, EVEC_SIZE, zero=False )

		eventArr[ TYPE ]      = self.Type
		eventArr[ PLAYEDBY ]  = self.PlayedBy
		eventArr[ RAISEAMT ]  = self.RaiseAmt
		eventArr[ BETTOTAL ]  = self.BetTotal
		eventArr[ IS_ALLIN ]  = self.Is_AllIn
		eventArr[ ALLINDIFF ] = self.AllInDiff
		eventArr[ DEALTO ]    = self.DealTo
		eventArr[ CDEALT ]    = self.CDealt

		return eventArr

	# DEPRECATED. Used to be called for generating unique identifiers from game histories.
	cdef str   GTString( self ): #noexcept:
		return str( NP( self.to_array() ).tobytes() )

	# Gives us a human-readable string for when we need to manually inspect event details
	cdef str   ShortString( self ): #noexcept:

		cdef uint2 cDealt = cyarr( (1,CVEC_SIZE), UINTSIZE, 'I' )
		cDealt[ 0 ] = CardVector( self.CDealt ) # Yeah it's dumb but this has to be 2d because reasons

		cdef str pStr = PLAYERCODES[ self.PlayedBy ],                                                                  \
				 rStr = str( self.RaiseAmt ).rjust( 4 ),                                                               \
				 bStr = str( self.BetTotal ).rjust( 4 ),                                                               \
				 aStr = str( self.Is_AllIn ).rjust( 4 ),                                                               \
				 dStr = str( self.AllInDiff ).rjust( 4 ),                                                              \
				 tStr = TYPENAMES[ self.Type ],                                                                        \
				 cStr = ''.join( PrettyCardStrings( cDealt ) )

		if (self.Type==PLAYERDEAL) and (self.CDealt==0): cStr = '∅∅'

		cdef tuple eData = ( tStr, pStr, rStr, bStr, aStr, dStr, self.DealTo, cStr )
		return '| |'.join( [f"{label}: {data}" for (label,data) in zip( EKEYS,eData )] )

	# Another inspection function, useful for printing events as part of game histories
	cdef str   ShorterString( self, uint stepNum, bint Include_Array=FALSE ): #noexcept:

		cdef uint2 cDealt = cyarr( (1,CVEC_SIZE), UINTSIZE, 'I' )
		cDealt[ 0 ] = CardVector( self.CDealt ) # Yeah it's dumb but this has to be 2d because reasons

		cdef str NOPE  = ' '*6,                                                                                       \
				 sStr  = str( stepNum ).rjust( 6 ),                                                                   \
				 tStr  = TYPENAMES[ self.Type ].center( 6 ),                                                          \
				 pStr  = PLAYERNAMES[ self.PlayedBy ].rjust( 6 )     if self.Type!=NULLEVENT else NOPE,               \
				 rStr  = str( self.RaiseAmt ).rjust( 6 )             if self.Type!=NULLEVENT else NOPE,               \
				 bStr  = str( self.BetTotal ).rjust( 6 )             if self.Type!=NULLEVENT else NOPE,               \
				 aStr  = str( self.Is_AllIn ).rjust( 6 )             if self.Type!=NULLEVENT else NOPE,               \
				 dStr  = str( self.AllInDiff ).rjust( 6 )            if self.Type!=NULLEVENT else NOPE,               \
				 dtStr = str( self.DealTo ).rjust( 6 )               if self.Type!=NULLEVENT else NOPE,               \
				 cStr  = PrettyCardStrings( cDealt,Compact=TRUE )[0] if self.Type!=NULLEVENT else NOPE

		# Blinds technically considered raises, adjust type str for readability if blind step
		if stepNum==5: 
			tStr = "SB".ljust( 6 )
		if stepNum==6: 
			tStr = "BB".ljust( 6 )

		# Is this a private deal with censored cards?
		cdef bint Redacted_Deal = (self.Type==PLAYERDEAL) and (self.CDealt==0)
		if Redacted_Deal:
			cStr = '∅∅'.center( 6 )

		cdef list eData = [ sStr, tStr, pStr, rStr, bStr, aStr, dStr, dtStr, cStr ]

		if not Include_Array: 
			return '| |'.join( eData ) + '| |'

		# If we need to print the raw event array, make it readable first
		cdef:
			list  arrList = []
			uint1 eArr    = self.to_array()
			uint  i
			str   iStr

		for i from 0 <= i < EVEC_SIZE:
			iStr = str( eArr[ i ] )

			if i not in [ RAISEAMT, BETTOTAL, ALLINDIFF ]:
				arrList.append( iStr )
			elif len( iStr )==3: 
				arrList.append( iStr )
			elif len( iStr )==2: 
				arrList.append( '0'+iStr )
			elif len( iStr )==1: 
				arrList.append( '00'+iStr )

		eData.append( (' '*6) + '[ ' + ' '.join( arrList ) + ' ]' )
		return '| |'.join( eData )

	cdef bint __EQ__( self, gameevent e ): #noexcept:

		cdef uint1 e1 = self.to_array(), e2 = e.to_array()
		cdef uint  i

		for i from 0 <= i < EVEC_SIZE:
			if e1[ i ] != e2[ i ]: 
				return FALSE

		return TRUE

	def __eq__( self, gameevent e ): 
		return self.__EQ__( e )


# Returns an array of non-raise actions, can be indexed using etypes
# Useful for building sets of available player actions
cdef uint2 AllNonRaises( uint player, uint callAmt=0, bint AllIn_Call=FALSE, uint callDiff=0 ): #noexcept:

	cdef uint2 nonRaises   = cyarr( (NON_RAISE_ETYPES+1, EVEC_SIZE), UINTSIZE, 'I' )
	cdef uint  p           = player

	nonRaises[ NULLEVENT ] = gameevent( NULLEVENT ).to_array()
	nonRaises[ FOLD ]  = gameevent( FOLD,  p ).to_array()
	nonRaises[ CHECK ] = gameevent( CHECK, p ).to_array()
	nonRaises[ CALL ]  = gameevent( CALL,  p, 
									betTotal=callAmt, Is_AllIn=AllIn_Call, allInDiff=callDiff ).to_array()

	return nonRaises

cdef inline bint  Is_Dealer_Action( uint1 e ): #noexcept:
	return ((e[ TYPE ]==PLAYERDEAL) or (e[ TYPE ]==BOARDDEAL))

cdef inline bint  Is_Null( uint1 e ): #noexcept:
	return e[ TYPE ]==NULLEVENT


# *-* # 