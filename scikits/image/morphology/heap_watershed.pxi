"""
Originally part of CellProfiler, code licensed under both GPL and BSD
licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
"""
import numpy as np
cimport numpy as np
cimport cython

cdef struct Heapitem:
    np.int32_t value
    np.int32_t age
    np.int32_t index

cdef inline int smaller(Heapitem *a, Heapitem *b):
    if a.value <> b.value:
      return a.value < b.value
    return a.age < b.age

include "heap_general.pxi"
