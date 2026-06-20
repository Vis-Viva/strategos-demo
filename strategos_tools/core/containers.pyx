#distutils: language = c
#cython: language_level 3
#cython: profile = False


from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memset


# ==================================================================================================
# This is just a bunch of lightweight, very fast custom containers equipped with specific 
# functionality we need downstream. There's nothing fancy here, these are bare-bones, manually
# managed numeric containers which exist to give us basic storage without incurring Python overhead.
# ==================================================================================================


cdef class vector_int:

	# No python-level entities used here, only C-level init needed
	def __cinit__( self, uint init_cap=0, int1 from_view=None, bint zero=FALSE ):

		if from_view is None:
			self.size     = 0
			self.capacity = init_cap if init_cap != 0 else 1

		elif from_view is not None:
			self.size     = from_view.shape[ 0 ]
			self.capacity = self.size 

		cdef uint cap   = self.capacity
		self._data      = <int *>malloc( cap * INTSIZE )
		self._data_view = <int[:cap]>self._data

		# Special init conditions: are we setting values from an existing memview or initializing to 0?
		if from_view is not None: self._data_view[:] = from_view
		elif zero:                self._data_view[:] = 0

	# Get min n such that a capacity of 2ⁿ will contain the vector's size; lets us do progressive cap growth
	cdef uint      __get_min_spanning_cap( self ): #noexcept:

		cdef uint n
		for n from 0 <= n <= 99:
			if 2**n > self.size: 
				return 2**n

		raise MemoryError( "Exactly what the fuck are you doing that you need a vector > 2^99 in size for???" )

	# Reallocates vector mem space to fit needed capacity
	cdef void        resize( self, uint newCap=0 ): #noexcept:
		self.capacity      = newCap if newCap != 0 else self.__get_min_spanning_cap()
		cdef int *newAlloc = <int *>realloc( self._data, self.capacity * INTSIZE )
		self._data         = newAlloc
		self._data_view    = <int[:self.capacity]>self._data

	# Just shrinks vector's allocated mem space to fit current size
	cdef void        shrink_wrap( self ): #noexcept:
		self.resize( newCap=self.size )

	cdef bint        contains( self, int item ): #noexcept:

		cdef uint n = self.size, i

		for i from 0 <= i < n:
			if self._data_view[ i ] == item: 
				return TRUE

		return FALSE

	cdef inline int  at( self, uint idx ): #noexcept:
		return self._data_view[ idx ]

	cdef uint        index_of( self, int item ): #noexcept:

		cdef uint n = self.size, idx

		for idx from 0 <= idx < n:
			if self._data_view[ idx ] == item:
				return idx

		raise IndexError( "Your princess is in another castle" )

	cdef void        set_data( self, int1 from_view ): #noexcept:

		cdef uint setSize = from_view.shape[ 0 ]

		if setSize > self.capacity: 
			self.resize( newCap=setSize )

		self._data_view[:] = from_view # DATA IS COPIED FOR SAFETY 

		self.size = setSize

	cdef void        append( self, int newElement ): #noexcept:

		self.size += 1

		if self.size > self.capacity: 
			self.resize()

		self._data_view[ self.size-1 ] = newElement

	cdef inline int  pop( self ): #noexcept:
		cdef int removedElement = self._data_view[ self.size-1 ]
		self.size -= 1
		return removedElement

	cdef inline void replace( self, uint at_idx, int newItem ): #noexcept:
		self._data_view[ at_idx ] = newItem

	cdef vector_int  copy( self ): #noexcept:
		return vector_int( from_view = self._data_view[:self.size] )

	# Downstream, we sometimes need a copy of a vector excluding a specific element
	cdef vector_int  without( self, int item ): #noexcept:

		cdef vector_int v = vector_int( init_cap=self.size )
		cdef uint n = self.size, i

		for i from 0 <= i < n:
			if self.at( i ) != item: 
				v.append( self.at( i ) )

		return v

	cdef int1        view( self ): #noexcept:
		return self._data_view[ :self.size ]

	def __dealloc__( self ): 
		if self._data is not NULL: 
			free( self._data )


cdef class vector_dbl:

	# No python-level entities used here, only C-level init needed
	def __cinit__( self, uint init_cap=0, dbl1 from_view=None, bint zero=FALSE ):

		if from_view is None:
			self.size     = 0
			self.capacity = init_cap if init_cap != 0 else 1
			
		elif from_view is not None:
			self.size     = from_view.shape[ 0 ]
			self.capacity = self.size 

		cdef uint cap   = self.capacity
		self._data      = <double *>malloc( cap * DBLSIZE )
		self._data_view = <double[:cap]>self._data

		# Special init conditions: are we setting values from an existing memview or initializing to 0?
		if from_view is not None: self._data_view[:] = from_view
		elif zero:                self._data_view[:] = 0

	# Get min n such that a capacity of 2ⁿ will contain the vector's size; lets us do progressive cap growth
	cdef uint        __get_min_spanning_cap( self ): #noexcept:

		cdef uint n
		for n from 0 <= n <= 99:
			if 2**n > self.size: 
				return 2**n

		raise MemoryError( "Exactly what the fuck are you doing that you need a vector > 2^99 in size for???" )

	# Reallocates vector mem space to fit needed capacity
	cdef void          resize( self, uint newCap=0 ): #noexcept:
		self.capacity         = newCap if newCap != 0 else self.__get_min_spanning_cap()
		cdef double *newAlloc = <double *>realloc( self._data, self.capacity * DBLSIZE )
		self._data            = newAlloc
		self._data_view       = <double[:self.capacity]>self._data

	# Just shrinks vector's allocated mem space to fit current size
	cdef void          shrink_wrap( self ): #noexcept:
		self.resize( newCap=self.size )

	cdef bint          contains( self, double item ): #noexcept:

		cdef uint n = self.size, i

		for i from 0 <= i < n:
			if self._data_view[ i ] == item: 
				return TRUE

		return FALSE

	cdef inline double at( self, uint idx ): #noexcept:
		return self._data_view[ idx ]

	cdef uint          index_of( self, double item ): #noexcept:

		cdef uint n = self.size, idx

		for idx from 0 <= idx < n:
			if self._data_view[ idx ] == item: 
				return idx

		raise IndexError( "Your princess is in another castle" )

	cdef void          set_data( self, dbl1 from_view ): #noexcept:

		cdef uint setSize = from_view.shape[ 0 ]

		if setSize > self.capacity: 
			self.resize( newCap=setSize )

		self._data_view[:] = from_view # DATA IS COPIED FOR SAFETY 

		self.size = setSize

	cdef void          append( self, double newElement ): #noexcept:

		self.size += 1

		if self.size > self.capacity: 
			self.resize()

		self._data_view[ self.size-1 ] = newElement

	cdef inline double pop( self ): #noexcept:
		cdef double removedElement = self._data_view[ self.size-1 ]
		self.size -= 1
		return removedElement

	cdef inline void   replace( self, uint at_idx, double newItem ): #noexcept:
		self._data_view[ at_idx ] = newItem

	cdef vector_dbl    copy( self ): #noexcept:
		return vector_dbl( from_view = self._data_view[:self.size] )

	# Downstream, we sometimes need a copy of a vector excluding a specific element
	cdef vector_dbl    without( self, double item ): #noexcept:

		cdef vector_dbl v = vector_dbl( init_cap=self.size )
		cdef uint n = self.size, i

		for i from 0 <= i < n:
			if self.at( i ) != item: 
				v.append( self.at( i ) )

		return v

	cdef dbl1          view( self ): #noexcept:
		return self._data_view[ :self.size ]

	def __dealloc__( self ): 
		if self._data is not NULL: 
			free( self._data )


cdef class vector_ll:

	# No python-level entities used here, only C-level init needed
	def __cinit__( self, uint init_cap=0, ll1 from_view=None, bint zero=FALSE ):

		if from_view is None:
			self.size     = 0
			self.capacity = init_cap if init_cap != 0 else 1

		elif from_view is not None:
			self.size     = from_view.shape[ 0 ]
			self.capacity = self.size 

		cdef uint cap   = self.capacity
		self._data      = <ll *>malloc( self.capacity * LLSIZE )
		self._data_view = <ll[:self.capacity]>self._data

		if from_view is not None: self._data_view[:] = from_view
		elif zero:                self._data_view[:] = 0

	# Get min n such that a capacity of 2ⁿ will contain the vector's size; lets us do progressive cap growth
	cdef uint      __get_min_spanning_cap( self ): #noexcept:

		cdef uint n
		for n from 0 <= n <= 99:
			if 2**n > self.size: return 2**n

		raise MemoryError( "Exactly what the fuck are you doing that you need a vector > 2^99 in size for???" )

	# Reallocates vector mem space to fit needed capacity
	cdef void        resize( self, uint newCap=0 ): #noexcept:
		self.capacity     = newCap if newCap != 0 else self.__get_min_spanning_cap()
		cdef ll *newAlloc = <ll *>realloc( self._data, self.capacity * LLSIZE )
		self._data        = newAlloc
		self._data_view   = <ll[:self.capacity]>self._data

	# Just shrinks vector's allocated mem space to fit current size
	cdef void        shrink_wrap( self ): #noexcept:
		self.resize( newCap=self.size )

	cdef bint        contains( self, ll item ): #noexcept:

		cdef uint n = self.size, i

		for i from 0 <= i < n:
			if self._data_view[ i ] == item: 
				return TRUE

		return FALSE

	cdef inline ll   at( self, uint idx ): #noexcept:
		return self._data_view[ idx ]

	cdef uint        index_of( self, ll item ): #noexcept:

		cdef uint n = self.size, idx

		for idx from 0 <= idx < n:
			if self._data_view[ idx ] == item: 
				return idx

		raise IndexError( "Your princess is in another castle" )

	cdef void        set_data( self, ll1 from_view ): #noexcept:

		cdef uint setSize = from_view.shape[ 0 ]

		if setSize > self.capacity: 
			self.resize( newCap=setSize )

		self._data_view[:] = from_view # DATA IS COPIED FOR SAFETY 

		self.size = setSize

	cdef void        append( self, ll newElement ): #noexcept:

		self.size+=1

		if self.size > self.capacity: 
			self.resize()

		self._data_view[ self.size-1 ] = newElement

	cdef inline ll   pop( self ): #noexcept:
		cdef ll removedElement = self._data_view[ self.size-1 ]
		self.size-=1
		return removedElement

	cdef inline void replace( self, uint at_idx, ll newItem ): #noexcept:
		self._data_view[ at_idx ] = newItem

	cdef vector_ll   copy( self ): #noexcept:
		return vector_ll( from_view = self._data_view[:self.size] )

	# Downstream, we sometimes need a copy of a vector excluding a specific element
	cdef vector_ll   without( self, ll item ): #noexcept:

		cdef vector_ll v = vector_ll( init_cap=self.size )
		cdef uint n = self.size, i

		for i from 0 <= i < n:
			if self.at( i ) != item: 
				v.append( self.at( i ) )

		return v

	cdef ll1         view( self ): #noexcept:
		return self._data_view[ :self.size ]

	def __dealloc__( self ): 
		if self._data is not NULL: 
			free( self._data )


cdef class vector_str:

	# Init C-level entities
	def __cinit__( self, uint init_cap=0 ):
		self.size     = 0
		self.capacity = init_cap if init_cap != 0 else 1
		self._data    = <void **>malloc( self.capacity * PTRSIZE )

	# Need a pylist to persist objects our stored pointers will point to; need normal init for this
	# Objects must be kept in this list so they stay alive & in place for pointers stored in _data.
	# NB this list is not intended to ever be used for access as pylist access is very inefficient.
	# For access, get ptr from _data and do <str> cast instead; this SHOULD bypass python overhead.
	def __init__( self, uint init_cap=0 ): 
		self.__dataList = []

	# Get min n such that a capacity of 2ⁿ will contain the vector's size; lets us do progressive cap growth
	cdef uint __get_min_spanning_cap( self ): #noexcept:

		cdef uint n
		for n from 0 <= n < 99:
			if 2**n > self.size: 
				return 2**n

		raise MemoryError( "Exactly what the fuck are you doing that you need a vector > 2^99 in size for???" )

	# Reallocates vector mem space to fit needed capacity
	cdef void   resize( self, uint newCap=0 ): #noexcept:
		self.capacity        = newCap if newCap != 0 else self.__get_min_spanning_cap()
		cdef void **newAlloc = <void **>realloc( self._data, self.capacity * PTRSIZE )
		self._data           = newAlloc

	# Just shrinks vector's allocated mem space to fit current size
	cdef void   shrink_wrap( self ): #noexcept:
		self.resize( newCap=self.size )

	cdef void   append( self, str newItem ): #noexcept:

		self.__dataList.append( newItem ) # Yeah it's slow but it's unavoidable

		self.size += 1

		if self.size > self.capacity: 
			self.resize()

		self._data[ self.size-1 ] = <void *>self.__dataList[ self.size-1 ]

	cdef str    at( self, uint idx ): #noexcept:
		return <str>(self._data[ idx ])

	def __dealloc__( self ): 
		if self._data is not NULL: 
			free( self._data )


cdef class matrix_int:
	
	# No python-level entities used here, only C-level init needed
	def __cinit__( self, uint nrows=0, uint ncols=0, int2 from_view=None ):

		cdef uint R = nrows if from_view is None else from_view.shape[ 0 ]
		cdef uint C = ncols if from_view is None else from_view.shape[ 1 ]

		self.nrows   = R
		self.ncols   = C
		self.rowsize = C
		self.size    = R*C

		self._data      = <int *>malloc( self.size * INTSIZE )
		self._data_view = <int[:R,:C]>self._data

		if from_view is not None: self._data_view[:] = from_view

	cdef void set_data( self, int2 from_view ): #noexcept:
		self._data_view[:] = from_view

	cdef int2 view( self ): #noexcept:
		return self._data_view[ :self.nrows, :self.ncols ]

	def __dealloc__( self ):
		if self._data is not NULL: 
			free( self._data )


cdef class matrix_flt:
	
	# No python-level entities used here, only C-level init needed
	def __cinit__( self, uint nrows=0, uint ncols=0, flt2 from_view=None ):

		cdef uint R = nrows if from_view is None else from_view.shape[ 0 ]
		cdef uint C = ncols if from_view is None else from_view.shape[ 1 ]

		self.nrows   = R
		self.ncols   = C
		self.rowsize = C
		self.size    = R*C

		self._data      = <float *>malloc( self.size * FLTSIZE )
		self._data_view = <float[:R,:C]>self._data

		if from_view is not None: 
			self._data_view[:] = from_view

	cdef void set_data( self, flt2 from_view ): #noexcept:
		self._data_view[:] = from_view

	cdef flt2 view( self ): #noexcept:
		return self._data_view[ :self.nrows, :self.ncols ]

	def     __dealloc__( self ):
		if self._data is not NULL: 
			free( self._data )


cdef class matrix_dbl:
	
	# No python-level entities used here, only C-level init needed
	def __cinit__( self, uint nrows=0, uint ncols=0, dbl2 from_view=None ):

		cdef uint R = nrows if from_view is None else from_view.shape[ 0 ]
		cdef uint C = ncols if from_view is None else from_view.shape[ 1 ]

		self.nrows   = R
		self.ncols   = C
		self.rowsize = C
		self.size    = R*C

		self._data      = <double *>malloc( self.size * DBLSIZE )
		self._data_view = <double[:R,:C]>self._data

		if from_view is not None: 
			self._data_view[:] = from_view

	cdef void set_data( self, dbl2 from_view ): #noexcept:
		self._data_view[:] = from_view

	cdef dbl2 view( self ): #noexcept:
		return self._data_view[ :self.nrows, :self.ncols ]

	def __dealloc__( self ):
		if self._data is not NULL: 
			free( self._data )


# Basically a 3d float array class; we treat this like a container which stores flt matrices.
cdef class MultiMat:
	
	# No python-level entities used here, only C-level init needed
	def __cinit__( self, flt3 from_view ):

		cdef uint M = from_view.shape[ 0 ], R = from_view.shape[ 1 ], C = from_view.shape[ 2 ]

		self.nmats   = M
		self.nrows   = R
		self.ncols   = C
		self.size    = M*R*C
		self.matsize = R*C  
		self.rowsize = C

		self._data         = <float *>malloc( self.size * FLTSIZE )
		self._data_view    = <float[ :M,:R,:C ]>self._data
		self._data_view[:] = from_view

	cdef void set_matrix( self, uint matIdx, flt2 from_data ): #noexcept:
		self._data_view[ matIdx ] = from_data

	cdef flt3 view( self ): #noexcept:
		return self._data_view[ :self.nmats, :self.nrows, :self.ncols ]

	cdef void replace( self, uint at_idx, matrix_flt newMat ): #noexcept:
		self._data_view[ at_idx ] = newMat.view()

	def __dealloc__( self ):
		if self._data is not NULL: 
			free( self._data )


# Lightweight, very fast custom long int key -> int value map. Equipped with specific functionality we need downstream.
cdef class index_map:
	
	# No python-level entities used here, only C-level init needed
	def __cinit__( self ):
		self.Keys    = vector_ll()
		self.Indices = vector_int()

	cdef int  at( self, ll key ): #noexcept:
		return self.Indices.at( self.Keys.index_of( key ) )

	cdef void append( self, ll newKey, int newIndex ): #noexcept:
		self.Keys.append( newKey )
		self.Indices.append( newIndex )

		
# *-* #