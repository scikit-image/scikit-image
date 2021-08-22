"""
Originally part of CellProfiler, code licensed under both GPL and BSD
licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
"""
cimport numpy as cnp


cdef struct Heapitem:
    cnp.float64_t value
    cnp.int32_t age
    Py_ssize_t index
    Py_ssize_t source


cdef inline int smaller(Heapitem *a, Heapitem *b) nogil:
    if a.value != b.value:
        return a.value < b.value
    return a.age < b.age


include "heap_general.pxi"
