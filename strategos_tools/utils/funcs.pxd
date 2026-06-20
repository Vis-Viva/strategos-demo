#distutils: language = c
#cython: language_level 3

cimport cython
from strategos_tools.core.CONSTS cimport *


# ==================================================================================================
# MEMVIEW MATH
# ==================================================================================================


# ----- 1D ARITHMETIC ----------------------------------------------------------

cdef float ArrMax1d( flt1 a ) #noexcept

cdef flt1  ArrClip1d( flt1 a, float minVal=* ) #noexcept

cdef float ArrSum1d( flt1 a ) #noexcept

cdef flt1  ArrMult1d( flt1 a, flt1 b ) #noexcept

cdef flt1  ArrScale1d( flt1 a, float x ) #noexcept


# ----- 2D ARITHMETIC ----------------------------------------------------------

cdef flt2  ArrAdd2d( flt2 a, flt2 b ) #noexcept

cdef flt2  ArrSub2d( flt2 a, flt2 b ) #noexcept

cdef flt2  ArrMult2d( flt2 a, flt2 b ) #noexcept

cdef flt2  ArrDiv2d( flt2 a, flt2 b ) #noexcept

cdef flt2  ArrEq2d( flt2 a, flt2 b ) #noexcept


# ----- 3D ARITHMETIC ----------------------------------------------------------

cdef flt3  ArrAdd3d( flt3 a, flt3 b ) #noexcept

cdef flt3  ArrSub3d( flt3 a, flt3 b ) #noexcept

cdef flt3  ArrMult3d( flt3 a, flt3 b ) #noexcept

cdef flt3  ArrDiv3d( flt3 a, flt3 b ) #noexcept

cdef flt3  ArrEq3d( flt3 a, flt3 b ) #noexcept


# ==================================================================================================
# NON-ARITHMETIC MEMVIEW FUNCS
# ==================================================================================================


cdef flt1  arange( uint stop, uint start=* ) #noexcept

cdef uint  MIN( uint1 a ) #noexcept

cdef uint  MAX( uint1 a ) #noexcept

cdef uint2 Has_Nonzero( flt2 arr, uint along_axis=* ) #noexcept

cdef flt2  NumNonzero( flt2 arr, uint along_axis=* ) #noexcept

cdef flt2  Unzero2d( flt2 a ) #noexcept

cdef flt3  Unzero3d( flt3 a ) #noexcept

cdef void  UnzeroAdvs( flt2 advSums ) #noexcept


# ==================================================================================================
# GENERAL UTIL FUNCS
# ==================================================================================================


cdef str   HMS( double totalSeconds ) #noexcept

cdef list  series( list data, int precision ) #noexcept

cdef str   progressbar( uint nCompleted, uint nTotal, uint indent=*, uint pbWidth=* ) #noexcept

cdef void _clear_current_line() #noexcept

cdef void _move_console_cursor_up( uint numLines ) #noexcept

cdef void _clear_prev_lines( uint numLines ) #noexcept
