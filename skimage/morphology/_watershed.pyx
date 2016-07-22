"""watershed.pyx - scithon implementation of guts of watershed

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
"""
import numpy as np
cimport numpy as cnp
cimport cython


ctypedef cnp.int32_t DTYPE_INT32_t
DTYPE_BOOL = np.bool
ctypedef cnp.int8_t DTYPE_BOOL_t


include "heap_watershed.pxi"


cdef extern from "math.h":
    double sqrt(double x)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
@cython.unraisable_tracebacks(False)
cdef inline double _euclid_dist(cnp.int32_t pt0, cnp.int32_t pt1,
                                cnp.int32_t[::1] strides):
    """Return the Euclidean distance between raveled points pt0 and pt1."""
    cdef float result = 0
    cdef float curr = 0
    for i in range(strides.shape[0]):
        curr = (pt0 // strides[i]) - (pt1 // strides[i])
        result += curr * curr
        pt0 = pt0 % strides[i]
        pt1 = pt1 % strides[i]
    return sqrt(result)


@cython.boundscheck(False)
def watershed(cnp.float64_t[::1] image,
              DTYPE_INT32_t[::1] marker_locations,
              DTYPE_INT32_t[::1] structure,
              DTYPE_BOOL_t[::1] mask,
              cnp.int32_t[::1] strides,
              cnp.float32_t compactness,
              DTYPE_INT32_t[::1] output):
    """Do heavy lifting of watershed algorithm

    Parameters
    ----------

    image - the flattened image pixels, converted to rank-order
    pq    - the priority queue, starts with the marked pixels
            the first element in each row is the image intensity
            the second element is the age at entry into the queue
            the third element is the index into the flattened image or labels
            the remaining elements are the coordinates of the point
    age   - the next age to assign to a pixel
    structure - a numpy int32 array containing the structuring elements
                that define nearest neighbors. For each row, the first
                element is the stride from the point to its neighbor
                in a flattened array. The remaining elements are the
                offsets from the point to its neighbor in the various
                dimensions
    mask  - numpy boolean (char) array indicating which pixels to consider
            and which to ignore. Also flattened.
    output - put the image labels in here
    """
    cdef Heapitem elem
    cdef Heapitem new_elem
    cdef Py_ssize_t nneighbors = structure.shape[0]
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t age = 1
    cdef Py_ssize_t index = 0

    cdef Heap *hp = <Heap *> heap_from_numpy2()

    for i in range(marker_locations.shape[0]):
        index = marker_locations[i]
        elem.value = image[index]
        elem.age = 0
        elem.index = index
        elem.source = index
        heappush(hp, &elem)

    while hp.items > 0:
        heappop(hp, &elem)
        if output[elem.index] and elem.index != elem.source:
            # non-marker, already visited from another neighbor
            continue
        output[elem.index] = output[elem.source]
        for i in range(nneighbors):
            # get the flattened address of the neighbor
            index = structure[i] + elem.index
            if output[index] or not mask[index]:
                # previously visited, masked, or border pixel
                continue

            age += 1
            new_elem.value = image[index]
            if compactness > 0:
                new_elem.value += (compactness *
                                   _euclid_dist(index, elem.source, strides))
            new_elem.age = age
            new_elem.index = index
            new_elem.source = elem.source
            
            heappush(hp, &new_elem)
    heap_done(hp)
