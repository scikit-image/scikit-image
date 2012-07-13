"""
`reconstruction_loop` originally part of CellProfiler, code licensed under both GPL and BSD licenses.

Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky

"""

from __future__ import division
import numpy as np

cimport numpy as np
cimport cython


@cython.boundscheck(False)
def reconstruction_loop(np.ndarray[dtype=np.uint32_t, ndim=1,
                                   negative_indices = False,
                                   mode = 'c'] avalues,
                        np.ndarray[dtype=np.int32_t, ndim=1,
                                   negative_indices = False,
                                   mode = 'c'] aprev,
                        np.ndarray[dtype=np.int32_t, ndim=1,
                                   negative_indices = False,
                                   mode = 'c'] anext,
                        np.ndarray[dtype=np.int32_t, ndim=1,
                                   negative_indices = False,
                                   mode = 'c'] astrides,
                        np.int32_t current,
                        int image_stride):
    """The inner loop for reconstruction"""
    cdef:
        np.int32_t neighbor
        np.uint32_t neighbor_value
        np.uint32_t current_value
        np.uint32_t mask_value
        np.int32_t link
        int i
        np.int32_t nprev
        np.int32_t nnext
        int nstrides = astrides.shape[0]
        np.uint32_t *values = <np.uint32_t *>(avalues.data)
        np.int32_t *prev = <np.int32_t *>(aprev.data)
        np.int32_t *next = <np.int32_t *>(anext.data)
        np.int32_t *strides = <np.int32_t *>(astrides.data)

    while current != -1:
        if current < image_stride:
            current_value = values[current]
            if current_value == 0:
                break
            for i in range(nstrides):
                neighbor = current + strides[i]
                neighbor_value = values[neighbor]
                # Only do neighbors less than the current value
                if neighbor_value < current_value:
                    mask_value = values[neighbor + image_stride]
                    # Only do neighbors less than the mask value
                    if neighbor_value < mask_value:
                        # Raise the neighbor to the mask value if
                        # the mask is less than current
                        if mask_value < current_value:
                            link = neighbor + image_stride
                            values[neighbor] = mask_value
                        else:
                            link = current
                            values[neighbor] = current_value
                        # unlink the neighbor
                        nprev = prev[neighbor]
                        nnext = next[neighbor]
                        next[nprev] = nnext
                        if nnext != -1:
                            prev[nnext] = nprev
                        # link the neighbor after the link
                        nnext = next[link]
                        next[neighbor] = nnext
                        prev[neighbor] = link
                        if nnext >= 0:
                            prev[nnext] = neighbor
                            next[link] = neighbor
        current = next[current]

