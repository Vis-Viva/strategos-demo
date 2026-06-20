#distutils: language = c
#cython: language_level 3


from strategos_tools.core.CONSTS cimport *


cdef class vector_int:

	cdef:
		uint  size, capacity
		int *_data
		int1 _data_view

	cdef uint     __get_min_spanning_cap( self ) #noexcept

	cdef void       resize( self, uint newCap=* ) #noexcept

	cdef void       shrink_wrap( self ) #noexcept

	cdef bint       contains( self, int item ) #noexcept

	cdef int        at( self, uint idx ) #noexcept

	cdef uint       index_of( self, int item ) #noexcept

	cdef void       set_data( self, int1 from_view ) #noexcept

	cdef void       append( self, int newElement ) #noexcept

	cdef int        pop( self ) #noexcept

	cdef void       replace( self, uint at_idx, int newItem ) #noexcept

	cdef vector_int copy( self ) #noexcept

	cdef vector_int without( self, int item ) #noexcept

	cdef int1       view( self ) #noexcept


cdef class vector_dbl:

	cdef:
		uint     size, capacity
		double *_data
		dbl1    _data_view

	cdef uint     __get_min_spanning_cap( self ) #noexcept

	cdef void       resize( self, uint newCap=* ) #noexcept

	cdef void       shrink_wrap( self ) #noexcept

	cdef bint       contains( self, double item ) #noexcept

	cdef double     at( self, uint idx ) #noexcept

	cdef uint       index_of( self, double item ) #noexcept

	cdef void       set_data( self, dbl1 from_view ) #noexcept

	cdef void       append( self, double newElement ) #noexcept

	cdef double     pop( self ) #noexcept

	cdef void       replace( self, uint at_idx, double newItem ) #noexcept

	cdef vector_dbl copy( self ) #noexcept

	cdef vector_dbl without( self, double item ) #noexcept

	cdef dbl1 view( self ) #noexcept


cdef class vector_ll:

	cdef:
		uint  size, capacity
		ll  *_data
		ll1  _data_view

	cdef uint    __get_min_spanning_cap( self ) #noexcept

	cdef void      resize( self, uint newCap=* ) #noexcept

	cdef void      shrink_wrap( self ) #noexcept

	cdef bint      contains( self, ll item ) #noexcept

	cdef ll        at( self, uint idx ) #noexcept

	cdef uint      index_of( self, ll item ) #noexcept

	cdef void      set_data( self, ll1 from_view ) #noexcept

	cdef void      append( self, ll newElement ) #noexcept

	cdef ll        pop( self ) #noexcept

	cdef void      replace( self, uint at_idx, ll newItem ) #noexcept

	cdef vector_ll copy( self ) #noexcept

	cdef vector_ll without( self, ll item ) #noexcept

	cdef ll1       view( self ) #noexcept


cdef class vector_str:

	cdef:
		uint    size, capacity
		void **_data
		list  __dataList

	cdef uint __get_min_spanning_cap( self ) #noexcept

	cdef void   resize( self, uint newCap=* ) #noexcept

	cdef void   shrink_wrap( self ) #noexcept

	cdef void   append( self, str newItem ) #noexcept

	cdef str    at( self, uint idx ) #noexcept


cdef class matrix_int:

	cdef:
		uint  size, nrows, ncols, rowsize
		int *_data
		int2 _data_view

	cdef void set_data( self, int2 from_view ) #noexcept

	cdef int2 view( self ) #noexcept


cdef class matrix_flt:

	cdef:
		uint    size, nrows, ncols, rowsize
		float *_data
		flt2   _data_view

	cdef void set_data( self, flt2 from_view ) #noexcept

	cdef flt2 view( self ) #noexcept


cdef class matrix_dbl:

	cdef:
		uint     size, nrows, ncols, rowsize
		double *_data
		dbl2    _data_view

	cdef void set_data( self, dbl2 from_view ) #noexcept

	cdef dbl2 view( self ) #noexcept


cdef class MultiMat:

	cdef:
		uint    size, nmats, matsize, nrows, rowsize, ncols
		float *_data
		flt3   _data_view

	cdef void set_matrix( self, uint matIdx, flt2 from_data ) #noexcept

	cdef flt3 view( self ) #noexcept

	cdef void replace( self, uint at_idx, matrix_flt newMat ) #noexcept


cdef class index_map:

	cdef vector_ll  Keys
	cdef vector_int Indices

	cdef int  at( self, ll key ) #noexcept

	cdef void append( self, ll newKey, int newIndex ) #noexcept

	
# *-* #