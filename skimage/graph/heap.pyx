# -*- python -*-

"""Cython implementation of a binary min heap.

Original author: Almar Klein
Modified by: Zachary Pincus

License: BSD

Copyright 2009 Almar Klein

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import division

# cython specific imports
import cython
from libc.stdlib cimport malloc, free

cdef extern from "pyport.h":
  double Py_HUGE_VAL

cdef VALUE_T inf = Py_HUGE_VAL

# this is handy
cdef inline INDEX_T index_min(INDEX_T a, INDEX_T b): return a if a <= b else b


cdef class BinaryHeap:
    """BinaryHeap(initial_capacity=128)

    A binary heap class that can store values and an integer reference.

    A binary heap is an object to store values in, optimized in such a way
    that the minimum (or maximum, but a minimum in this implementation)
    value can be found in O(log2(N)) time. In this implementation, a reference
    value (a single integer) can also be stored with each value.

    Use the methods push() and pop() to put in or extract values.
    In C, use the corresponding push_fast() and pop_fast().

    Parameters
    ----------
    initial_capacity : int
        Estimate of the size of the heap, if known in advance. (In any case,
        the heap will dynamically grow and shrink as required, though never
        below the `initial_capacity`.)

    Attributes
    ----------
    count : int
        The number of values in the heap
    levels : int
        The number of levels in the binary heap (see Notes below). The values
        are stored in the last level, so 2**levels is the capacity of the
        heap before another resize is required.
    min_levels : int
        The minimum number of levels in the heap (relates to the
        `initial_capacity` parameter.)

    Notes
    -----
    This implementation stores the binary heap in an array twice as long as
    the number of elements in the heap. The array is structured in levels,
    starting at level 0 with a single value, doubling the amount of values in
    each level. The final level contains the actual values, the level before
    it contains the pairwise minimum values. The level before that contains
    the pairwise minimum values of that level, etc. Take a look at this
    illustration:

    level: 0 11 2222 33333333 4444444444444444
    index: 0 12 3456 78901234 5678901234567890
                        1          2         3

     The actual values are stored in level 4. The minimum value of position 15
    and 16 is stored in position 7. min(17,18)->8, min(7,8)->3, min(3,4)->1.
    When adding a value, only the path to the top has to be updated, which
    takesO(log2(N)) time.

     The advantage of this implementation relative to more common
    implementations that swap values when pushing to the array is that data
    only needs to be swapped once when an element is removed. This means that
    keeping an array of references along with the values is very inexpensive.
    Th disadvantage is that if you pop the minimum value, the tree has to be
    traced from top to bottom and back. So if you only want values and no
    references, this implementation will probably be slower. If you need
    references (and maybe cross references to be kept up to date) this
    implementation will be faster.
    """

    ## Basic methods
    # The following lines are always "inlined", but documented here for
    # clarity:
    #
    # To calculate the start index of a certain level:
    # 2**l-1 # LevelStart
    # Note that in inner loops, this may also be represented as (1<<l)-1,
    # because code of the form x**y goes via the python pow operations and
    # can thus be a bit slower.
    #
    # To calculate the corresponding ABSOLUTE index at the next level:
    # i*2+1 # CalcNextAbs
    #
    # To calculate the corresponding ABSOLUTE index at the previous level:
    # (i-1)/2 # CalcPrevAbs
    #
    # To calculate the capacity at a certain level:
    # 2**l
    def __cinit__(self, INDEX_T initial_capacity=128, *args, **kws):
        # calc levels from the default capacity
        cdef LEVELS_T levels = 0
        while 2**levels < initial_capacity:
            levels += 1
        # set levels
        self.min_levels = self.levels = levels

        # we start with 0 values
        self.count = 0

        # allocate arrays
        cdef INDEX_T number = 2**self.levels
        self._values = <VALUE_T *>malloc( 2*number * sizeof(VALUE_T))
        self._references = <REFERENCE_T *>malloc(number * sizeof(REFERENCE_T))

    def __init__(self, INDEX_T initial_capacity=128):
        """__init__(initial_capacity=128)

        Class constructor.

        Takes an optional parameter 'initial_capacity' so that
        if the required heap capacity is known or can be estimated in advance,
        there will need to be fewer resize operations on the heap."""
        if self._values is NULL or self._references is NULL:
          raise MemoryError()
        self.reset()

    def reset(self):
        """reset()

        Reset the heap to default, empty state.
        """
        cdef INDEX_T number = 2**self.levels
        cdef INDEX_T i
        cdef VALUE_T *values = self._values
        for i in range(number*2):
            values[i] = inf


    def __dealloc__(self):
        if self._values is not NULL:
            free(self._values)
        if self._references is not NULL:
            free(self._references)

    def __str__(self):
        s = ''
        for level in range(1,self.levels+1):
            i0 = 2**level-1 # LevelStart
            s+= 'level %i: ' % level
            for i in range(i0,i0+2**level):
                s += '%g, ' % self._values[i]
            s = s[:-1] + '\n'
        return s


    ## C Maintanance methods

    cdef void _add_or_remove_level(self, LEVELS_T add_or_remove):
        # init indexing ints
        cdef INDEX_T i, i1, i2, n

        # new amount of levels
        cdef LEVELS_T new_levels = self.levels + add_or_remove

        # allocate new arrays
        cdef INDEX_T number = 2**new_levels
        cdef VALUE_T *values
        cdef REFERENCE_T *references
        values = <VALUE_T *>malloc(number*2 * sizeof(VALUE_T))
        references = <REFERENCE_T *>malloc(number * sizeof(REFERENCE_T))

        # init arrays
        for i in range(number*2):
            values[i] = inf
        for i in range(number):
            references[i] = -1

        # copy data
        cdef VALUE_T *old_values = self._values
        cdef REFERENCE_T *old_references = self._references
        if self.count:
            i1 = 2**new_levels-1 # LevelStart
            i2 = 2**self.levels-1 # LevelStart
            n = index_min(2**new_levels, 2**self.levels)
            for i in range(n):
                values[i1+i] = old_values[i2+i]
            for i in range(n):
                references[i] = old_references[i]

        # make current
        free(self._values)
        free(self._references)
        self._values = values
        self._references = references

        # we need a full update
        self.levels = new_levels
        self._update()


    cdef void _update(self):
        """Update the full tree from the bottom up.
        This should be done after resizing. """

        # shorter name for values
        cdef VALUE_T *values = self._values

        # Note that i represents an absolute index here
        cdef INDEX_T i0, i, ii, n
        cdef LEVELS_T level

        # track tree
        for level in range(self.levels,1,-1):
            i0 = (1 << level) - 1 #2**level-1 = LevelStart
            n = i0 + 1 #2**level
            for i in range(i0,i0+n,2):
                ii = (i-1)//2 # CalcPrevAbs
                if values[i] < values[i+1]:
                    values[ii] = values[i]
                else:
                    values[ii] = values[i+1]


    cdef void _update_one(self, INDEX_T i):
        """Update the tree for one value."""

        # shorter name for values
        cdef VALUE_T *values = self._values

        # make index uneven
        if i % 2==0:
            i = i-1

        # track tree
        cdef INDEX_T ii
        cdef LEVELS_T level
        for level in range(self.levels,1,-1):
            ii = (i-1)//2 # CalcPrevAbs

            # test
            if values[i] < values[i+1]:
                values[ii] = values[i]
            else:
                values[ii] = values[i+1]
            # next
            if ii % 2:
                i = ii
            else:
                i = ii-1


    cdef void _remove(self, INDEX_T i1):
        """Remove a value from the heap. By index."""

        cdef LEVELS_T levels = self.levels
        cdef INDEX_T count = self.count
        # get indices
        cdef INDEX_T i0 = (1 << levels) - 1  #2**self.levels - 1 # LevelStart
        cdef INDEX_T i2 = i0 + count - 1

        # get relative indices
        cdef INDEX_T r1 = i1 - i0
        cdef INDEX_T r2 = count - 1

        cdef VALUE_T *values = self._values
        cdef REFERENCE_T *references = self._references

        # swap with last
        values[i1] = values[i2]
        references[r1] = references[r2]

        # make last Null
        values[i2] = inf

        # update
        self.count -= 1
        count -= 1
        if (levels>self.min_levels) & (count < (1 << (levels-2))):
            self._add_or_remove_level(-1)
        else:
            self._update_one(i1)
            self._update_one(i2)


    ## C Public methods

    cdef INDEX_T push_fast(self, VALUE_T value, REFERENCE_T reference):
        """The c-method for fast pushing.

        Returns the index relative to the start of the last level in the heap."""
        # We need to resize if currently it just fits.
        cdef LEVELS_T levels = self.levels
        cdef INDEX_T count = self.count
        if count >= (1 << levels):#2**self.levels:
            self._add_or_remove_level(+1)
            levels += 1

        # insert new value
        cdef INDEX_T i = ((1 << levels) - 1) + count # LevelStart + n
        self._values[i] = value
        self._references[count] = reference

        # update
        self.count += 1
        self._update_one(i)

        # return
        return count


    cdef VALUE_T pop_fast(self):
        """The c-method for fast popping.

        Returns the minimum value. The reference is put in self._popped_ref"""

        # shorter name for values
        cdef VALUE_T *values = self._values

        # init index. start at 1 because we start in level 1
        cdef LEVELS_T level
        cdef INDEX_T i = 1
        cdef LEVELS_T levels = self.levels
        # search tree (using absolute indices)
        for level in range(1, levels):
            if values[i] <= values[i+1]:
                i = i*2+1 # CalcNextAbs
            else:
                i = (i+1)*2+1 # CalcNextAbs

        # select best one in last level
        if values[i] <= values[i+1]:
            i = i
        else:
            i = i+1

        # get values
        cdef INDEX_T ir = i - ((1 << levels) - 1) #(2**self.levels-1) # LevelStart
        cdef VALUE_T value = values[i]
        self._popped_ref = self._references[ir]

        # remove it
        if self.count:
            self._remove(i)

        # return
        return value


    ## Python Public methods (that do not need to be VERY fast)

    def push(self, VALUE_T value, REFERENCE_T reference=-1):
        """push(value, reference=-1)

        Append a value to the heap, with optional reference.

        Parameters
        ----------
        value : float
            Value to push onto the heap
        reference : int, optional
            Reference to associate with the given value.
        """
        self.push_fast(value, reference)


    def min_val(self):
        """min_val()

        Get the minimum value on the heap.

        Returns only the value, and does not remove it from the heap.
        """
        # shorter name for values
        cdef VALUE_T *values = self._values

        # select best one in last level
        if values[1] < values[2]:
            return values[1]
        else:
            return values[2]


    def values(self):
        """values()

        Get the values in the heap as a list.
        """
        out = []
        cdef INDEX_T i, i0
        i0 = 2**self.levels-1  # LevelStart
        for i in range(self.count):
            out.append(self._values[i0+i])
        return out


    def references(self):
        """references()

        Get the references in the heap as a list.
        """
        out = []
        cdef INDEX_T i
        for i in range(self.count):
            out.append(self._references[i])
        return out


    def pop(self):
        """pop()

        Get the minimum value and remove it from the list.

        Returns
        -------
        value : float
        reference : int
            If no reference was provided, -1 is returned here.

        Raises
        ------
        IndexError
            On attempt to pop from an empty heap
        """
        if self.count == 0:
          raise IndexError('pop from an empty heap')
        value = self.pop_fast()
        ref = self._popped_ref
        return value, ref



cdef class FastUpdateBinaryHeap(BinaryHeap):
    """FastUpdateBinaryHeap(initial_capacity=128, max_reference=None)

    Binary heap that allows the value of a reference to be updated quickly.

    This heap class keeps cross-references so that the value associated with a
    given reference can be quickly queried (O(1) time) or updated (O(log2(N))
    time). This is ideal for pathfinding algorithms that implement some
    variant of Dijkstra's algorithm.

    Parameters
    ----------
    initial_capacity : int
        Estimate of the size of the heap, if known in advance. (In any case,
        the heap will dynamically grow and shrink as required, though never
        below the `initial_capacity`.)

    max_reference : int, optional
        Largest reference value that might be pushed to the heap. (Pushing a
        larger value will result in an error.) If no value is provided,
        `1-initial_capacity` will be used. For the cross-reference index to
        work, all references must be in the range [0, max_reference];
        references pushed outside of that range will not be added to the heap.
        The cross-references are kept as a 1-d array of length
        `max_reference+1', so memory use of this heap is effectively
        O(max_reference)

    Attributes
    ----------
    count : int
        The number of values in the heap
    levels : int
        The number of levels in the binary heap (see Notes below). The values
        are stored in the last level, so 2**levels is the capacity of the
        heap before another resize is required.
    min_levels : int
        The minimum number of levels in the heap (relates to the
        `initial_capacity` parameter.)
    max_reference : int
        The provided or calculated maximum allowed reference value.

    Notes
    -----
    The cross-references map data[reference]->internalindex, such that the
    value corresponding to a given reference can be found efficiently. This
    can be queried with the value_of() method.

    A special method, push_if_lower() is provided that will update the heap if
    the given reference is not in the heap, or if it is and the provided value
    is lower than the current value in the heap. This is again useful for
    pathfinding algorithms.
    """
    def __cinit__(self, INDEX_T initial_capacity=128, max_reference=None):
      if max_reference is None:
        max_reference = initial_capacity - 1
      self.max_reference = max_reference
      self._crossref = <INDEX_T *>malloc((self.max_reference+1) * \
                                      sizeof(INDEX_T))

    def __init__(self, INDEX_T initial_capacity=128, max_reference=None):
        """__init__(initial_capacity=128, max_reference=None)

        Class constructor.
        """
        # below will call self.reset
        BinaryHeap.__init__(self, initial_capacity)


    def __dealloc__(self):
        if self._crossref is not NULL:
            free(self._crossref)

    def reset(self):
        """reset()

        Reset the heap to default, empty state.
        """
        BinaryHeap.reset(self)
        # set default values of crossrefs
        cdef INDEX_T i
        for i in range(self.max_reference+1):
            self._crossref[i] = -1


    cdef void _remove(self, INDEX_T i1):
        """ Remove a value from the heap. By index. """
        cdef LEVELS_T levels = self.levels
        cdef INDEX_T count = self.count

        # get indices
        cdef INDEX_T i0 = (1 << levels) - 1  #2**self.levels - 1 # LevelStart
        cdef INDEX_T i2 = i0 + count - 1

        # get relative indices
        cdef INDEX_T r1 = i1 - i0
        cdef INDEX_T r2 = count - 1

        cdef VALUE_T *values = self._values
        cdef REFERENCE_T *references = self._references
        cdef INDEX_T *crossref = self._crossref

        # update cross reference
        crossref[references[r2]]=r1
        crossref[references[r1]]=-1  # disable removed item

        # swap with last
        values[i1] = values[i2]
        references[r1] = references[r2]

        # make last Null
        values[i2] = inf

        # update
        self.count -= 1
        count -= 1
        if (levels > self.min_levels) & (count < (1 << (levels-2))):
            self._add_or_remove_level(-1)
        else:
            self._update_one(i1)
            self._update_one(i2)


    cdef INDEX_T push_fast(self, VALUE_T value, REFERENCE_T reference):
        """The c method for fast pushing.

        If the reference is already present, will update its value, otherwise
        will append it.

        If -1 is returned, the provided reference was out-of-bounds and no
        value was pushed to the heap."""
        if not (0 <= reference <= self.max_reference):
          return -1

        # init variable to store the index-in-the-heap
        cdef INDEX_T i

        # Reference is the index in the array where MCP is applied to.
        # Find the index-in-the-heap using the crossref array.
        cdef INDEX_T ir = self._crossref[reference]

        if ir != -1:
            # update
            i = (1 << self.levels) - 1 + ir
            self._values[i] = value
            self._update_one(i)
            return ir

        # if not updated: append normally and store reference
        ir = BinaryHeap.push_fast(self, value, reference)
        self._crossref[reference] = ir
        return ir

    cdef INDEX_T push_if_lower_fast(self, VALUE_T value, REFERENCE_T reference):
        """If the reference is already present, will update its value ONLY if
        the new value is lower than the old one. If the reference is not
        present, this append it. If a value was appended, self._pushed is
        set to 1.

        If -1 is returned, the provided reference was out-of-bounds and no
        value was pushed to the heap.
        """
        if not (0 <= reference <= self.max_reference):
            return -1

        # init variable to store the index-in-the-heap
        cdef INDEX_T i

        # Reference is the index in the array where MCP is applied to.
        # Find the index-in-the-heap using the crossref array.
        cdef INDEX_T ir = self._crossref[reference]
        cdef VALUE_T *values = self._values
        self._pushed = 1
        if ir != -1:
            # update
            i = (1 << self.levels) - 1 + ir
            if values[i] > value:
                values[i] = value
                self._update_one(i)
            else:
                self._pushed = 0
            return ir

        # if not updated: append normally and store reference
        ir = BinaryHeap.push_fast(self, value, reference)
        self._crossref[reference] = ir
        return ir


    cdef VALUE_T value_of_fast(self, REFERENCE_T reference):
        """Return the value corresponding to the given reference. If inf
        is returned, the reference may be invalid: check the _invaild_ref
        field in this case."""

        if not (0 <= reference <= self.max_reference):
            self.invalid_ref = 1
            return inf

        # init variable to store the index-in-the-heap
        cdef INDEX_T i

        # Reference is the index in the array where MCP is applied to.
        # Find the index-in-the-heap using the crossref array.
        cdef INDEX_T ir = self._crossref[reference]
        self._invalid_ref = 0
        if ir == -1:
            self._invalid_ref = 1
            return inf
        i = (1 << self.levels) - 1 + ir
        return self._values[i]


    def push(self, double value, int reference):
        """push(value, reference)

        Append/update a value in the heap.

        Parameters
        ----------
        value : float
        reference : int
            If the reference is already present in the array, the value for
            that reference will be updated, otherwise the (value, reference)
            pair will be added to the heap.

        Raises
        ------
        ValueError
            On pushing a reference outside the range [0, max_reference].
        """
        if self.push_fast(value, reference) == -1:
            raise ValueError("reference outside of range [0, max_reference]")

    def push_if_lower(self, double value, int reference):
        """push_if_lower(value, reference)

        Append/update a value in the heap if the extant value is lower.

        If the reference is already in the heap, update only of the new value
        is lower than the current one. If the reference is not present, the
        value will always be pushed to the heap.

        Parameters
        ----------
        value : float
        reference : int
            If the reference is already present in the array, the value for
            that reference will be updated, otherwise the (value, reference)
            pair will be added to the heap.

        Returns
        -------
        pushed : bool
            True if an append/update occurred, False if otherwise.

        Raises
        ------
        ValueError
            On pushing a reference outside the range [0, max_reference].
        """
        if self.push_if_lower_fast(value, reference) == -1:
          raise ValueError("reference outside of range [0, max_reference]")
        return self._pushed == 1

    def value_of(self, int reference):
        """value_of(reference)

        Get the value corresponding to a given reference.

        Parameters
        ----------
        reference : int
            A reference already pushed to the heap.

        Returns
        -------
        value : float

        Raises
        ------
        ValueError
            On querying a reference outside the range [0, max_reference], or
            not already pushed to the heap.
        """
        value = self.value_of_fast(reference)
        if self._invalid_ref:
            raise ValueError('invalid reference')
        return value

    def cross_references(self):
        """Get the cross references in the heap as a list."""
        out = []
        cdef INDEX_T i
        for i in range(self.max_reference+1):
            out.append( self._crossref[i] )
        return out
