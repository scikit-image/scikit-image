""" This is the definition file for heap.pyx.
It contains the definitions of the heap classes, such that
other cython modules can "cimport heap" and thus use the
C versions of pop(), push(), and value_of(): pop_fast(), push_fast() and
value_of_fast()
"""

# determine datatypes for heap
ctypedef double VALUE_T
ctypedef Py_ssize_t REFERENCE_T
ctypedef REFERENCE_T INDEX_T
ctypedef unsigned char BOOL_T
ctypedef unsigned char LEVELS_T

cdef class BinaryHeap:
    cdef readonly INDEX_T count
    cdef readonly LEVELS_T levels, min_levels
    cdef VALUE_T *_values
    cdef REFERENCE_T *_references
    cdef REFERENCE_T _popped_ref

    cdef void _add_or_remove_level(self, LEVELS_T add_or_remove) nogil
    cdef void _update(self) nogil
    cdef void _update_one(self, INDEX_T i) nogil
    cdef void _remove(self, INDEX_T i) nogil

    cdef INDEX_T push_fast(self, VALUE_T value, REFERENCE_T reference) nogil
    cdef VALUE_T pop_fast(self) nogil

cdef class FastUpdateBinaryHeap(BinaryHeap):
    cdef readonly REFERENCE_T max_reference
    cdef INDEX_T *_crossref
    cdef BOOL_T _invalid_ref
    cdef BOOL_T _pushed

    cdef VALUE_T value_of_fast(self, REFERENCE_T reference)
    cdef INDEX_T push_if_lower_fast(self, VALUE_T value,
                                    REFERENCE_T reference) nogil
