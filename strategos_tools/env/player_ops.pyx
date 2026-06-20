#distutils: language = c
#cython: language_level 3
#cython: profile = False


from cpython.array cimport clone as pyarr


# ==================================================================================================
# Players are nothing special, we don't use any kind of special class for them - they're literally 
# just integer IDs. This module just provides a handful of useful utility functions which do 
# player-related operations commonly used for game logic.
# ==================================================================================================


# Returns array of the given player's opponents
cdef uint1       OpponentsOf( uint player ): #noexcept:

	cdef uint  oppIdx    = 0, p
	cdef uint1 opponents = pyarr( ARR_TMPLT_I, NUM_PLAYERS-1, zero=False )

	# Start at 1 because DEALER (aka 0) is everyone's friend :)
	for p from 1 <= p <= NUM_PLAYERS:
		if p != player:
			opponents[ oppIdx ] = p
			oppIdx += 1

	return opponents[ :oppIdx ]

cdef inline bint Are_Opponents( uint p1, uint p2 ): #noexcept:
	return FALSE if ( (p1==DEALER) or (p2==DEALER) or (p1==p2) ) else TRUE

# Useful for game logic which needs to find the next player to act
cdef inline uint NextNonDealer( uint p ): #noexcept:
	return p+1 if p+1 <= NUM_PLAYERS else 1

# Useful for when we need to find who was the previous player to act
cdef inline uint PrevNonDealer( uint p ): #noexcept:
	return p-1 if p-1 > 0 else NUM_PLAYERS


# *-* #