"""watershed.pyx - cython implementation of guts of watershed
"""
from libc.math cimport sqrt
from .._shared.fused_numerics cimport np_anyint
import numpy as np

cimport numpy as cnp
cimport cython
cnp.import_array()

ctypedef cnp.int8_t DTYPE_BOOL_t


include "heap_watershed.pxi"


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
@cython.unraisable_tracebacks(False)
cdef inline cnp.float64_t _euclid_dist(Py_ssize_t pt0, Py_ssize_t pt1,
                                       cnp.intp_t[::1] strides) noexcept nogil:
    """Return the Euclidean distance between raveled points pt0 and pt1."""
    cdef cnp.float64_t result = 0
    cdef cnp.float64_t curr = 0
    for i in range(strides.shape[0]):
        curr = (pt0 // strides[i]) - (pt1 // strides[i])
        result += curr * curr
        pt0 = pt0 % strides[i]
        pt1 = pt1 % strides[i]
    return sqrt(result)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.unraisable_tracebacks(False)
cdef inline DTYPE_BOOL_t _diff_neighbors(np_anyint[::1] output,
                                         cnp.intp_t[::1] structure,
                                         DTYPE_BOOL_t[::1] mask,
                                         Py_ssize_t index,
                                         np_anyint label,
                                         ) noexcept nogil:
    """
    Return ``True`` and set ``mask[index]`` to ``False`` if the neighbors of
    ``index`` (as given by the offsets in ``structure``) have more than one
    distinct nonzero label.
    """
    cdef:
        Py_ssize_t i, neighbor_index
        np_anyint neighbor_label
        Py_ssize_t nneighbors = structure.shape[0]

    if not mask[index]:
        return True

    for i in range(nneighbors):
        neighbor_index = structure[i] + index
        if mask[neighbor_index]:  # neighbor not a watershed line
            neighbor_label = output[neighbor_index]
            if neighbor_label and neighbor_label != label:
                mask[index] = False
                return True
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
def watershed_raveled(cnp.float64_t[::1] image,
                      cnp.intp_t[::1] marker_locations,
                      cnp.intp_t[::1] structure,
                      DTYPE_BOOL_t[::1] mask,
                      cnp.intp_t[::1] strides,
                      cnp.float64_t compactness,
                      np_anyint[::1] output,
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
    cdef Py_ssize_t neighbor_index = 0
    cdef DTYPE_BOOL_t compact = (compactness > 0)
    cdef cnp.float64_t neg_inf = -np.inf

    cdef Heap *hp = <Heap *> heap_from_numpy2()

    with nogil:
        for i in range(marker_locations.shape[0]):
            index = marker_locations[i]
            elem.value = neg_inf
            elem.age = 0
            elem.index = index
            elem.source = index
            heappush(hp, &elem)

        while hp.items > 0:
            heappop(hp, &elem)

            if compact or wsl:
                # in the compact case, we need to label pixels as they come off
                # the heap, because the same pixel can be pushed twice, *and* the
                # later push can have lower cost because of the compactness.
                #
                # In the case of preserving watershed lines, a similar argument
                # applies: we can only observe that all neighbors have been labeled
                # as the pixel comes off the heap. Trying to do so at push time
                # is a bug.
                if output[elem.index] and elem.index != elem.source:
                    # non-marker, already visited from another neighbor
                    continue

                # when `wsl` is `True`, label is only set for pixels without a neighbor of different label
                # NOTE: `_diff_neighbors` sets `mask[elem.index]` to `False` if
                #        neighbor has different label
                if compact or not _diff_neighbors(output, structure, mask, elem.index, output[elem.source]):
                    output[elem.index] = output[elem.source]

            for i in range(nneighbors):
                # get the flattened address of the neighbor
                neighbor_index = structure[i] + elem.index

                if not mask[neighbor_index]:
                    # this branch includes basin boundaries, aka watershed lines
                    # neighbor is not in mask
                    continue

                if output[neighbor_index]:
                    # pre-labeled neighbor is not added to the queue.
                    continue

                age += 1
                new_elem.value = image[neighbor_index]
                if compact:
                    new_elem.value += (compactness *
                                       _euclid_dist(neighbor_index, elem.source,
                                                    strides))
                elif not wsl:
                    # in the simplest watershed case (no compactness and no
                    # watershed lines), we can label a pixel at the time that
                    # we push it onto the heap, because it can't be reached with
                    # lower cost later.
                    # This results in a very significant performance gain, see:
                    # https://github.com/scikit-image/scikit-image/issues/2636
                    output[neighbor_index] = output[elem.index]
                new_elem.age = age
                new_elem.index = neighbor_index
                new_elem.source = elem.source

                # watershed cost of moving to neighbor is at least the cost of
                # its own neighboring pixel
                if new_elem.value < elem.value:
                    new_elem.value = elem.value

                heappush(hp, &new_elem)

    heap_done(hp)
