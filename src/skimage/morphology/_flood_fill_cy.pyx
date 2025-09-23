#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

"""Cython code used in _flood_fill.py."""

cimport numpy as cnp
cnp.import_array()

# Must be defined to use QueueWithHistory
ctypedef Py_ssize_t QueueItem

include "../morphology/_queue_with_history.pxi"


ctypedef fused dtype_t:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.float32_t
    cnp.float64_t


# Definition of flag values used for `flags` in _flood_fill & _fill_plateau
cdef:
    # Border value - do not cross!
    unsigned char BORDER = 2
    # Part of the flood fill
    unsigned char FILL = 1
    # Not checked yet
    unsigned char UNKNOWN = 0


cpdef inline void _flood_fill_equal(dtype_t[::1] image,
                                    unsigned char[::1] flags,
                                    Py_ssize_t[::1] neighbor_offsets,
                                    Py_ssize_t start_index,
                                    dtype_t seed_value):
    """Find connected areas to fill, requiring strict equality.

    Parameters
    ----------
    image : ndarray, one-dimensional
        The raveled view of a n-dimensional array.
    flags : ndarray, one-dimensional
        An array of flags that is used to store the state of each pixel during
        evaluation.
    neighbor_offsets : ndarray
        A one-dimensional array that contains the offsets to find the
        connected neighbors for any index in `image`.
    start_index : int
        Start position for the flood-fill.
    seed_value :
        Value of ``image[start_index]``.
    """
    cdef:
        QueueWithHistory queue
        QueueItem current_index, neighbor

    with nogil:
        # Initialize the queue
        queue_init(&queue, 64)
        try:
            queue_push(&queue, &start_index)
            flags[start_index] = FILL
            # Break loop if all queued positions were evaluated
            while queue_pop(&queue, &current_index):
                # Look at all neighboring samples
                for i in range(neighbor_offsets.shape[0]):
                    neighbor = current_index + neighbor_offsets[i]

                    # Shortcut if neighbor is already part of fill
                    if flags[neighbor] == UNKNOWN:
                        if image[neighbor] == seed_value:
                            # Neighbor is in fill; check its neighbors too.
                            flags[neighbor] = FILL
                            queue_push(&queue, &neighbor)
        finally:
            # Ensure memory released
            queue_exit(&queue)


cpdef inline void _flood_fill_tolerance(dtype_t[::1] image,
                                        unsigned char[::1] flags,
                                        Py_ssize_t[::1] neighbor_offsets,
                                        Py_ssize_t start_index,
                                        dtype_t seed_value,
                                        dtype_t low_tol,
                                        dtype_t high_tol):
    """Find connected areas to fill, within a tolerance.

    Parameters
    ----------
    image : ndarray, one-dimensional
        The raveled view of a n-dimensional array.
    flags : ndarray, one-dimensional
        An array of flags that is used to store the state of each pixel during
        evaluation.
    neighbor_offsets : ndarray
        A one-dimensional array that contains the offsets to find the
        connected neighbors for any index in `image`.
    start_index : int
        Start position for the flood-fill.
    seed_value :
        Value of ``image[start_index]``.
    low_tol :
        Lower limit for tolerance comparison.
    high_tol :
        Upper limit for tolerance comparison.
    """
    cdef:
        QueueWithHistory queue
        QueueItem current_index, neighbor

    with nogil:
        # Initialize the queue and push start position
        queue_init(&queue, 64)
        try:
            queue_push(&queue, &start_index)
            flags[start_index] = FILL
            # Break loop if all queued positions were evaluated
            while queue_pop(&queue, &current_index):
                # Look at all neighboring samples
                for i in range(neighbor_offsets.shape[0]):
                    neighbor = current_index + neighbor_offsets[i]

                    # Only do comparisons on points not (yet) part of fill
                    if flags[neighbor] == UNKNOWN:
                        if low_tol <= image[neighbor] <= high_tol:
                            # Neighbor is in fill; check its neighbors too.
                            flags[neighbor] = FILL
                            queue_push(&queue, &neighbor)
        finally:
            # Ensure memory released
            queue_exit(&queue)
