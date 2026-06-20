# distutils: language = c
# cython: language_level 3
# cython: profile = False


from cpython.array cimport clone as pyarr
from cython.view   cimport array as cyarr
from libc.limits   cimport UINT_MAX
from libc.math     cimport INFINITY

import numpy as np
from numpy import asarray as NP, expand_dims as newaxis
from numpy import float64 as f32, float32 as f32, intc, uintc


# ==================================================================================================
# Just a bunch of standalone general utility functions that make doing other things 
# cleaner/easier/faster. Lots of this is just here as wrappers to make arr ops syntax more
# readable. A bunch of the array arithmetic does copyless creation of numpy views so we can 
# use their optimized inbuilt math. Some stuff uses manual loops where it's proven to be faster.
# ==================================================================================================


# ==================================================================================================
# MEMVIEW MATH
# ==================================================================================================


# ----- 1D ARITHMETIC ----------------------------------------------------------

cdef float ArrMax1d( flt1 a ): #noexcept:
	cdef uint  aSize = a.shape[ 0 ], i
	cdef float aMax  = -INFINITY
	for i from 0 <= i < aSize:
		if a[ i ] > aMax: 
			aMax = a[ i ]
	return aMax

cdef flt1  ArrClip1d( flt1 a, float minVal=0 ): #noexcept:
	cdef uint aSize    = a.shape[ 0 ], i
	cdef flt1 aClipped = cyarr( (aSize,), FLTSIZE, 'f' )
	for i from 0 <= i < aSize: 
		aClipped[ i ] = a[ i ] if a[ i ]>minVal else minVal
	return aClipped

cdef float ArrSum1d( flt1 a ): #noexcept:
	cdef float s=0
	cdef uint  aSize = a.shape[ 0 ], i
	for i from 0 <= i < aSize: 
		s+=a[ i ]
	return s

cdef flt1  ArrMult1d( flt1 a, flt1 b ): #noexcept:
	return NP( a,dtype=f32 ) * NP( b,dtype=f32 )

cdef flt1  ArrScale1d( flt1 a, float x ): #noexcept:
	cdef uint arrsize = a.shape[ 0 ], i
	cdef flt1 aScaled = pyarr( ARR_TMPLT_f, arrsize, zero=False )
	for i from 0 <= i < arrsize: 
		aScaled[ i ] = x * a[ i ]
	return aScaled


# ----- 2D ARITHMETIC ----------------------------------------------------------

cdef flt2  ArrAdd2d( flt2 a, flt2 b ): #noexcept:
	return NP( a,dtype=f32 ) + NP( b,dtype=f32 )

cdef flt2  ArrSub2d( flt2 a, flt2 b ): #noexcept:
	return NP( a,dtype=f32 ) - NP( b,dtype=f32 )

cdef flt2  ArrMult2d( flt2 a, flt2 b ): #noexcept:
	return NP( a,dtype=f32 ) * NP( b,dtype=f32 )

# ASSUMES b IS ENTIRELY NONZERO
cdef flt2  ArrDiv2d( flt2 a, flt2 b ): #noexcept:
	return NP( a,dtype=f32 ) / NP( b,dtype=f32 )

cdef flt2  ArrEq2d( flt2 a, flt2 b ): #noexcept:
	return NP( NP(a)==NP(b),dtype=f32 )


# ----- 3D ARITHMETIC ----------------------------------------------------------

cdef flt3  ArrAdd3d( flt3 a, flt3 b ): #noexcept:
	return NP( a,dtype=f32 ) + NP( b,dtype=f32 )

cdef flt3  ArrSub3d( flt3 a, flt3 b ): #noexcept:
	return NP( a,dtype=f32 ) - NP( b,dtype=f32 )

cdef flt3  ArrMult3d( flt3 a, flt3 b ): #noexcept:
	return NP( a,dtype=f32 ) * NP( b,dtype=f32 )

# ASSUMES b IS ENTIRELY NONZERO
cdef flt3  ArrDiv3d( flt3 a, flt3 b ): #noexcept:
	return NP( a,dtype=f32 ) / NP( b,dtype=f32 )

cdef flt3  ArrEq3d( flt3 a, flt3 b ): #noexcept:
	return NP( NP(a)==NP(b),dtype=f32 )


# ==================================================================================================
# NON-ARITHMETIC MEMVIEW FUNCS
# ==================================================================================================


cdef flt1  arange( uint stop, uint start=1 ): #noexcept:

	cdef uint size = stop-start+1, i=0, n 
	cdef flt1 a    = pyarr( ARR_TMPLT_f, size, zero=False )

	for n from start <= n <= stop:
		a[ i ] = n
		i+=1

	return a

# Avoids incurring bloat from python's min function
cdef uint  MIN( uint1 a ): #noexcept:
	
	cdef uint arrmin = UINT_MAX, arrsize = a.shape[ 0 ], i
	for i from 0 <= i < arrsize:
		if a[ i ] < arrmin: 
			arrmin = a[ i ]
		
	return arrmin

# Avoids incurring bloat from python's max function
cdef uint  MAX( uint1 a ): #noexcept:

	cdef uint arrmax = 0, arrsize = a.shape[ 0 ], i
	for i from 0 <= i < arrsize:
		if a[ i ] > arrmax: 
			arrmax = a[ i ]

	return arrmax

# output[i] is 1 if input has any nonzero elements in the i-th element of the specified axis.
# Designed for a specific purpose for EstimatorOps; keeps an extra empty dim for this reason. 
cdef uint2 Has_Nonzero( flt2 arr, uint along_axis=0 ): #noexcept:
	return NP( (NP( arr )>0).any( axis=along_axis, keepdims=TRUE ),dtype=uintc )

# Counts the number of nonzero elements along specified axis. 
# Designed for a specific purpose for EstimatorOps; keeps an extra empty dim for this reason.
cdef flt2  NumNonzero( flt2 arr, uint along_axis=0 ): #noexcept:
	return NP( np.count_nonzero( arr, axis=along_axis, keepdims=1 ),dtype=f32 )

# Returns copy of a with 1s replacing 0s; useful for avoiding 0-div errors where 0s in a can be ignored
cdef flt2  Unzero2d( flt2 a ): #noexcept:

	cdef flt2 aUnzeroed = a.copy()
	cdef uint i,j
	for i from 0 <= i < a.shape[ 0 ]:
		for j from 0 <= j < a.shape[ 1 ]:
			if a[ i,j ]==0.0: 
				aUnzeroed[ i,j ]=1.0

	return aUnzeroed

# Returns copy of a with 1s replacing 0s; useful for avoiding 0-div errors where 0s in a can be ignored
cdef flt3  Unzero3d( flt3 a ): #noexcept:

	cdef flt3 aUnzeroed = a.copy()
	cdef uint i,j,k
	for i from 0 <= i < a.shape[ 0 ]:
		for j from 0 <= j < a.shape[ 1 ]:
			for k from 0 <= k < a.shape[ 2 ]:
				if a[ i,j,k ]==0.0: 
					aUnzeroed[ i,j,k ]=1.0

	return aUnzeroed

# Specific helper util for EstimatorOps to avoid 0-div errors
cdef void UnzeroAdvs( flt2 advSums ): #noexcept:
	cdef uint nI = advSums.shape[ 0 ], I
	for I from 0 <= I < nI:
		if advSums[ I,0 ]==0: 
			advSums[ I,0 ] = 1


# ==================================================================================================
# GENERAL UTIL FUNCS
# ==================================================================================================


# Turns a duration in seconds into a string of form {H}h{M}m{S}s
cdef str   HMS( double totalSeconds ): #noexcept:
	return f"{(<int>totalSeconds)//3600}h{((<int>totalSeconds)%3600)//60}m{(<int>totalSeconds)%60}s"

# Returns data as list[float] with supplied precision, input data can be any numeric type. Not fast.
cdef list  series( list data, int precision ): #noexcept:
	return [ float( f"{x:.{precision}f}" ) for x in data ]

# Gee I wonder what this could possibly do
cdef str   progressbar( uint nCompleted, uint nTotal, uint indent=0, uint pbWidth=100 ): #noexcept:

	cdef:
		float PB_SCALE    = pbWidth / 100,                                                                             \
			  percentDone = (nCompleted / nTotal) * 100
		uint  nBlocks     = <uint>(percentDone * PB_SCALE),                                                            \
			  nBlanks     = pbWidth - nBlocks
		str   pbBlocks    = "█" * nBlocks,                                                                             \
			  pbBlanks    = " " * nBlanks

	return ('\t'*indent) + "|" + pbBlocks + pbBlanks + "|" + f" {percentDone:.2f}%"

cdef void  _clear_current_line(): #noexcept:
	print( LINE_CLEAR, end='\r' )

# Moves the print position up by numLines of console output
cdef void  _move_console_cursor_up( uint numLines ): #noexcept:
	print( LINE_UP*numLines, end='\r' )

# Clears previous numLines of console output, use to overwrite prev lines of output
cdef void  _clear_prev_lines( uint numLines ): #noexcept:

	cdef uint line
	for line from 1 <= line < numLines:
		_move_console_cursor_up( 1 )
		_clear_current_line()


# ==================================================================================================
# PYTHON INFERFACE
# ==================================================================================================


def hms( totalSeconds ): 
	return HMS( totalSeconds )
	
def progress_bar( nCompleted, nTotal, indent=0, pbWidth=100 ): 
	return progressbar( nCompleted, nTotal, indent, pbWidth )
	
def clear_current_line():
	_clear_current_line()

def move_console_cursor_up( numLines ):
	_move_console_cursor_up( numLines )

def clear_prev_lines( numLines ):
	_clear_prev_lines( numLines )

		
# *-* # 