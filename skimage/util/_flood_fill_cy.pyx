#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

"""Cython code used in extrema.py."""

cimport numpy as cnp


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


# Definition of flag values used for `flags` in _local_maxima & _fill_plateau
cdef:
    # Part of the flood fill
    unsigned char FILL = 2
    # Checked already, not part of fill
    unsigned char NOT_FILL = 1
    # Not checked yet
    unsigned char UNKNOWN = 0


def _flood_fill(dtype_t[::1] image not None,
                unsigned char[::1] flags,
                Py_ssize_t[::1] neighbor_offsets not None,
                Py_ssize_t ravelled_seed_idx,
                dtype_t seed_value,
                unsigned char do_tol,
                dtype_t high_tol,
                dtype_t low_tol):
    """Find the region to be filled.

    Inner function to `flood_fill` that detects all connected points equal (or
    within tolerance) of the initial seed point.  The result is stored inplace 
    inside `flags` as the value "FILL".

    Parameters
    ----------
    image : ndarray, one-dimensional
        The raveled view of a n-dimensional array.
    flags : ndarray
        An array of flags that is used to store the state of each pixel during
        evaluation and is MODIFIED INPLACE. Initially, pixels that border the
        image edge must be marked as "BORDER_INDEX" while all other pixels
        should be marked with "EXCLUDED".
    neighbor_offsets : ndarray
        A one-dimensional array that contains the offsets to find the
        connected neighbors for any index in `image`.
    ravelled_seed_idx : int
        Ravelled seed index for flood filling.
    do_tol : unsigned char
        If 0, only exactly equal connected points are filled and `tolerance` is
        ignored (fastest).  Otherwise, points within `tolerance` of start value
        will also be filled (inclusive).
    high_tol :
        Upper limit for tolerance comparison (ignored if `do_tol` is 0).
    low_tol :
        Lower limit for tolerance comparison (ignored if `do_tol` is 0).
    """
    cdef:
        QueueWithHistory queue
        unsigned char last_dim_in_neighbors

    with nogil:
        # Initialize a buffer used to queue positions while evaluating each
        # potential maximum (flagged with 2)
        queue_init(&queue, 64)
        try:
            # Conduct flood fill from this point - with or without tolerance
            if do_tol == 0:
                _flood_fill_do_equal(image, flags, neighbor_offsets, &queue,
                    ravelled_seed_idx, seed_value)
            else:
                _flood_fill_do_tol(image, flags, neighbor_offsets, &queue,
                    ravelled_seed_idx, seed_value, high_tol, low_tol)
        finally:
            # Ensure that memory is released again
            queue_exit(&queue)


cdef inline void _flood_fill_do_equal(
        dtype_t[::1] image, unsigned char[::1] flags,
        Py_ssize_t[::1] neighbor_offsets, QueueWithHistory* queue_ptr,
        Py_ssize_t start_index, dtype_t seed_value) nogil:
    """Fill connected areas with 1, requiring strict equality.
    
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
    queue_ptr :
        Pointer to initialized queue.
    start_index : int
        Start position for the flood-fill.
    seed_value : 
        Value of `image[start_index]`.
    """
    cdef:
        QueueItem current_index, neighbor

    # Queue start position after clearing the buffer
    queue_clear(queue_ptr)
    queue_push(queue_ptr, &start_index)
    flags[start_index] = FILL

    # Break loop if all queued positions were evaluated
    while queue_pop(queue_ptr, &current_index):
        # Look at all neighboring samples
        for i in range(neighbor_offsets.shape[0]):
            neighbor = current_index + neighbor_offsets[i]

            # Indexing sanity check
            if neighbor < 0:
                continue
            if neighbor >= image.shape[0]:
                continue

            # Shortcut if neighbor is already part of fill
            if flags[neighbor] == UNKNOWN:
                if image[neighbor] == seed_value:
                # Neighbor is part of fill - but must check its neighbors too.
                    flags[neighbor] = FILL
                    queue_push(queue_ptr, &neighbor)
                else:
                    # Do not check this point again
                    flags[neighbor] = NOT_FILL


cdef inline void _flood_fill_do_tol(
        dtype_t[::1] image, unsigned char[::1] flags,
        Py_ssize_t[::1] neighbor_offsets, QueueWithHistory* queue_ptr,
        Py_ssize_t start_index, dtype_t seed_value, dtype_t high_tol,
        dtype_t low_tol) nogil:
    """Fill connected areas with 1, within a tolerance.
    
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
    queue_ptr :
        Pointer to initialized queue.
    start_index : int
        Start position for the flood-fill.
    seed_value : 
        Value of `image[start_index]`.
    high_tol :
        Upper limit for tolerance comparison.
    low_tol :
        Lower limit for tolerance comparison.
    """
    cdef:
        QueueItem current_index, neighbor

    # Queue start position after clearing the buffer
    queue_clear(queue_ptr)
    queue_push(queue_ptr, &start_index)
    flags[start_index] = FILL

    # Break loop if all queued positions were evaluated
    while queue_pop(queue_ptr, &current_index):
        # Look at all neighboring samples
        for i in range(neighbor_offsets.shape[0]):
            neighbor = current_index + neighbor_offsets[i]

            # Indexing sanity check
            if neighbor < 0:
                continue
            if neighbor >= image.shape[0]:
                continue
            
            # Only do comparisons on points not (yet) part of fill 
            if flags[neighbor] == UNKNOWN:
                if image[neighbor] <= high_tol:
                    if image[neighbor] >= low_tol:
                        # Neighbor is in fill; must check its neighbors too.
                        flags[neighbor] = FILL
                        queue_push(queue_ptr, &neighbor)
                    else:
                        # Do not check this point again
                        flags[neighbor] = NOT_FILL
                else:
                    # Do not check this point again
                    flags[neighbor] = NOT_FILL
