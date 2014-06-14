"""watershed.pyx - scithon implementation of guts of watershed

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
"""
import numpy as np
cimport numpy as np
cimport cython


ctypedef np.int32_t DTYPE_INT32_t
DTYPE_BOOL = np.bool
ctypedef np.int8_t DTYPE_BOOL_t


include "heap_watershed.pxi"


@cython.boundscheck(False)
def watershed(DTYPE_INT32_t[::1] image,
              DTYPE_INT32_t[:, ::1] pq,
              Py_ssize_t age,
              DTYPE_INT32_t[:, ::1] structure,
              DTYPE_BOOL_t[::1] mask,
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
    cdef Py_ssize_t index = 0
    cdef Py_ssize_t old_index = 0
    cdef Py_ssize_t max_index = image.shape[0]

    cdef Heap *hp = <Heap *> heap_from_numpy2()

    for i in range(pq.shape[0]):
        elem.value = pq[i, 0]
        elem.age = pq[i, 1]
        elem.index = pq[i, 2]
        heappush(hp, &elem)

    while hp.items > 0:
        #
        # Pop off an item to work on
        #
        heappop(hp, &elem)
        ####################################################
        # loop through each of the structuring elements
        #
        old_index = elem.index
        for i in range(nneighbors):
            # get the flattened address of the neighbor
            index = structure[i, 0] + old_index
            if index < 0 or index >= max_index or output[index] or \
                    not mask[index]:
                continue

            new_elem.value = image[index]
            new_elem.age = elem.age + 1
            new_elem.index = index
            age += 1
            output[index] = output[old_index]
            #
            # Push the neighbor onto the heap to work on it later
            #
            heappush(hp, &new_elem)
    heap_done(hp)
