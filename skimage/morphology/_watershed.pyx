"""watershed.pyx - scithon implementation of guts of watershed

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
"""
import numpy as np
from libc.math cimport sqrt

cimport numpy as cnp
cimport cython


ctypedef cnp.int32_t DTYPE_INT32_t
ctypedef cnp.int8_t DTYPE_BOOL_t


include "heap_watershed.pxi"


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
@cython.unraisable_tracebacks(False)
cdef inline double _euclid_dist(cnp.int32_t pt0, cnp.int32_t pt1,
                                cnp.int32_t[::1] strides):
    """Return the Euclidean distance between raveled points pt0 and pt1."""
    cdef double result = 0
    cdef double curr = 0
    for i in range(strides.shape[0]):
        curr = (pt0 // strides[i]) - (pt1 // strides[i])
        result += curr * curr
        pt0 = pt0 % strides[i]
        pt1 = pt1 % strides[i]
    return sqrt(result)


@cython.boundscheck(False)
def watershed_raveled(cnp.float64_t[::1] image,
                      DTYPE_INT32_t[::1] marker_locations,
                      DTYPE_INT32_t[::1] structure,
                      DTYPE_BOOL_t[::1] mask,
                      cnp.int32_t[::1] strides,
                      cnp.double_t compactness,
                      DTYPE_INT32_t[::1] output,
                      DTYPE_BOOL_t wsl):
    """Perform watershed algorithm using a raveled image and neighborhood.

    Parameters
    ----------

    image : array of float
        The flattened image pixels.
    marker_locations : array of int
        The raveled coordinates of the initial markers (aka seeds) for the
        watershed. NOTE: these should *all* point to nonzero entries in the
        output, or the algorithm will never terminate and blow up your memory!
    structure : array of int
        A list of coordinate offsets to compute the raveled coordinates of each
        neighbor from the raveled coordinates of the current pixel.
    mask : array of int
        An array of the same shape as `image` where each pixel contains a
        nonzero value if it is to be considered for flooding with watershed,
        zero otherwise. NOTE: it is *essential* that the border pixels (those
        with neighbors falling outside the volume) are all set to zero, or
        segfaults could occur.
    strides : array of int
        An array representing the number of steps to move along each dimension.
        This is used in computing the Euclidean distance between raveled
        indices.
    compactness : float
        A value greater than 0 implements the compact watershed algorithm
        (see .py file).
    output : array of int
        The output array, which must already contain nonzero entries at all the
        seed locations.
    wsl : bool
        Parameter indicating whether the watershed line is calculated.
        If wsl is set to True, the watershed line is calculated.
    """
    cdef Heapitem elem
    cdef Heapitem new_elem
    cdef Py_ssize_t nneighbors = structure.shape[0]
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t age = 1
    cdef Py_ssize_t index = 0
    cdef DTYPE_INT32_t wsl_label = -1
    cdef DTYPE_BOOL_t compact = (compactness > 0)

    cdef Heap *hp = <Heap *> heap_from_numpy2()

    for i in range(marker_locations.shape[0]):
        index = marker_locations[i]
        elem.value = image[index]
        elem.age = 0
        elem.index = index
        elem.source = index
        heappush(hp, &elem)
        if wsl and wsl_label >= output[index]:
            wsl_label = output[index] - 1

    while hp.items > 0:
        heappop(hp, &elem)

        # this can happen if the same pixel entered the queue
        # several times before being processed.
        if wsl and output[elem.index] == wsl_label:
            # wsl labels are not propagated.
            continue

        if compact or wsl:
            if output[elem.index] and elem.index != elem.source:
                # non-marker, already visited from another neighbor
                continue
            output[elem.index] = output[elem.source]

        for i in range(nneighbors):
            # get the flattened address of the neighbor
            neighbor_index = structure[i] + elem.index

            if not mask[neighbor_index]:
                # neighbor is not in mask
                continue

            if wsl and output[neighbor_index] == wsl_label:
                continue

            if output[neighbor_index]:
                # neighbor has a label (but not wsl_label):
                # the neighbor is not added to the queue.
                if wsl:
                    # if the label of the neighbor is different
                    # from the label of the pixel taken from the queue,
                    # the latter takes the WSL label.
                    if output[neighbor_index] != output[elem.index]:
                        output[elem.index] = wsl_label
                continue

            age += 1
            new_elem.value = image[neighbor_index]
            if compact:
                new_elem.value += (compactness *
                                   _euclid_dist(neighbor_index, elem.source,
                                                strides))
            elif not wsl:
                output[neighbor_index] = output[elem.index]
            new_elem.age = age
            new_elem.index = neighbor_index
            new_elem.source = elem.source

            heappush(hp, &new_elem)

    heap_done(hp)
