#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

"""Cython code used in extrema.py."""

import numpy as np
cimport numpy as cnp
from libc.math cimport pow


# Must be defined to use QueueWithHistory
ctypedef Py_ssize_t QueueItem

include "_queue_with_history.pxi"


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
    # First or last index in a dimension
    unsigned char BORDER_INDEX = 3
    # Potentially part of a maximum
    unsigned char CANDIDATE = 2
    # Index was queued (flood-fill) and might still be part of maximum OR
    # when evaluation is complete this flag value marks definite local maxima
    unsigned char QUEUED_CANDIDATE = 1
    # None of the above is true
    unsigned char NOT_MAXIMUM = 0


def _local_maxima(dtype_t[::1] image not None,
                  unsigned char[::1] flags,
                  Py_ssize_t[::1] neighbor_offsets not None):
    """Detect local maxima in n-dimensional array.

    Inner function to `local_maxima` that detects all local maxima (including
    plateaus) in the image. The result is stored inplace inside `flags` with
    the value of "QUEUED_CANDIDATE".

    Parameters
    ----------
    image : ndarray, one-dimensional
        The raveled view of a n-dimensional array.
    flags : ndarray
        An array of flags that is used to store the state of each pixel during
        evaluation and is MODIFIED INPLACE. Initially, pixels that border the
        image edge must be marked as "BORDER_INDEX" while all other pixels
        should be marked with "NOT_MAXIMUM". Modified in place.
    neighbor_offsets : ndarray
        A one-dimensional array that contains the offsets to find the
        connected neighbors for any index in `image`.
    """
    cdef:
        QueueWithHistory queue
        unsigned char last_dim_in_neighbors

    # This needs the GIL
    last_dim_in_neighbors = -1 in neighbor_offsets and 1 in neighbor_offsets
    with nogil:
        if last_dim_in_neighbors:
            # If adjacent pixels in the last dimension are part of the
            # neighborhood, the number of candidates which have to be evaluated
            # in the second algorithmic step `_fill_plateau` can be reduced
            # ahead of time with this function
            _mark_candidates_in_last_dimension(image, flags)
        else:
            # Otherwise simply mark all pixels (except border) as candidates
            _mark_candidates_all(flags)

        # Initialize a buffer used to queue positions while evaluating each
        # potential maximum (flagged with 2)
        queue_init(&queue, 64)
        try:
            for i in range(image.shape[0]):
                if flags[i] == CANDIDATE:
                    # Index is potentially part of a maximum:
                    # Find all samples which are part of the plateau and fill
                    # with 0 or 1 depending on whether it's a true maximum
                    _fill_plateau(image, flags, neighbor_offsets, &queue, i)
        finally:
            # Ensure that memory is released again
            queue_exit(&queue)


cdef inline void _mark_candidates_in_last_dimension(
        dtype_t[::1] image, unsigned char[::1] flags) nogil:
    """Mark local maxima in last dimension.
    
    This function considers only the last dimension of the image and marks 
    pixels with the "CANDIDATE" flag if it is a local maximum. 
    
    Parameters
    ----------
    image :
        The raveled view of a n-dimensional array.
    flags :
        An array of flags that is used to store the state of each pixel during
        evaluation. Modified in place.
    
    Notes
    -----
    By evaluating this necessary but not sufficient condition first, usually a
    significant amount of pixels can be rejected without having to evaluate the
    entire neighborhood of their plateau. This can reduces the number of 
    candidates that need to be evaluated with the more expensive flood-fill 
    performed in `_fill_plateaus`.
    
    However this is only possible if the adjacent pixels in the last dimension
    are part of the defined neighborhood (see argument `neighbor_offsets`
    in `_local_maxima`).
    """
    cdef Py_ssize_t i, i_ahead
    i = 1
    while i < image.shape[0]:
        if image[i - 1] < image[i] and flags[i] != BORDER_INDEX:
            # Potential maximum (in last dimension) is found, find
            # other edge of current plateau or "edge of dimension"
            i_ahead = i + 1
            while (
                image[i] == image[i_ahead] and
                flags[i_ahead] != BORDER_INDEX
            ):
                i_ahead += 1
            if image[i] > image[i_ahead]:
                # Found local maximum (in one dimension), mark all
                # parts of the plateau as potential maximum
                flags[i:i_ahead] = CANDIDATE
            i = i_ahead
        else:
            i += 1


cdef inline void _mark_candidates_all(unsigned char[::1] flags) nogil:
    """Mark all pixels as potential maxima, exclude border pixels.
    
    This function marks pixels with the "CANDIDATE" flag if they aren't the
    first or last index in any dimension (not flagged with "BORDER_INDEX"). 
    
    Parameters
    ----------
    flags :
        An array of flags that is used to store the state of each pixel during
        evaluation. Modified in place.
    """
    cdef Py_ssize_t i = 1
    while i < flags.shape[0]:
        if flags[i] != BORDER_INDEX:
            flags[i] = CANDIDATE
        i += 1


cdef inline void _fill_plateau(
        dtype_t[::1] image, unsigned char[::1] flags,
        Py_ssize_t[::1] neighbor_offsets, QueueWithHistory* queue_ptr,
        Py_ssize_t start_index) nogil:
    """Fill with 1 if plateau is local maximum else with 0.
    
    Parameters
    ----------
    image :
        The raveled view of a n-dimensional array.
    flags :
        An array of flags that is used to store the state of each pixel during
        evaluation. Modified in place.
    neighbor_offsets :
        A one-dimensional array that contains the offsets to find the
        connected neighbors for any index in `image`.
    queue_ptr :
        Pointer to initialized queue.
    start_index :
        Start position for the flood-fill.
    """
    cdef:
        dtype_t height
        unsigned char is_maximum
        QueueItem current_index, neighbor

    height = image[start_index]  # Height of the evaluated plateau
    is_maximum = 1 # Boolean flag, initially assume true

    # Queue start position after clearing the buffer
    # which might have been used already
    queue_clear(queue_ptr)
    queue_push(queue_ptr, &start_index)
    flags[start_index] = QUEUED_CANDIDATE

    # Break loop if all queued positions were evaluated
    while queue_pop(queue_ptr, &current_index):
        # Look at all neighboring samples
        for i in range(neighbor_offsets.shape[0]):
            neighbor = current_index + neighbor_offsets[i]

            if image[neighbor] == height:
                # Value is part of plateau
                if flags[neighbor] == BORDER_INDEX:
                    # Plateau touches border and can't be maximum
                    is_maximum = 0
                elif flags[neighbor] != QUEUED_CANDIDATE:
                    # Index wasn't queued already, do so now
                    queue_push(queue_ptr, &neighbor)
                    flags[neighbor] = QUEUED_CANDIDATE

            elif image[neighbor] > height:
                # Current plateau can't be maximum because it borders a
                # larger one
                is_maximum = 0

    if not is_maximum:
        queue_restore(queue_ptr)
        # Initial guess was wrong -> flag as NOT_MAXIMUM
        while queue_pop(queue_ptr, &neighbor):
            flags[neighbor] = NOT_MAXIMUM


cdef inline cnp.float64_t _sq_euclidean_distance(
    Py_ssize_t p1,
    Py_ssize_t p2,
    Py_ssize_t[::1] unravel_factors
) nogil:
    """Calculate the squared euclidean distance between two points in a raveled array.

    Parameters
    ----------
    p1, p2 :
        Two raveled coordinates (indices to a raveled array). 
    unravel_factors :
        An array of factors for all dimensions except the first one. E.g. if the
        unraveled array has the shape ``(1, 2, 3, 4)`` this should be 
        ``np.array([2 * 3 * 4, 3 * 4, 4])``.

    Returns
    -------
    sq_distance :
        The squared euclidean distance between `p1` and `p2`.
    """
    cdef:
        Py_ssize_t i, div1, div2, mod1, mod2
        cnp.float64_t sq_distance
    sq_distance = 0
    mod1 = p1
    mod2 = p2
    for i in range(unravel_factors.shape[0]):
        div1 = mod1 // unravel_factors[i]
        div2 = mod2 // unravel_factors[i]
        mod1 %= unravel_factors[i]
        mod2 %= unravel_factors[i]
        sq_distance += pow(div1 - div2, 2)
    sq_distance += pow(mod1 - mod2, 2)
    return sq_distance


def _remove_close_maxima(
    unsigned char[::1] maxima,
    tuple shape,
    Py_ssize_t[::1] neighbor_offsets,
    Py_ssize_t[::1] priority,
    cnp.float64_t minimal_distance,
):
    """Remove maxima which are to close to maxima with larger values.

    Parameters
    ----------
    maxima : ndarray, one-dimensional
        A raveled boolean array indicating the positions of local maxima which
        can be reshaped with `shape`. Modified in place.
    shape : tuple
        A tuple indicating the shape of the unraveled `maxima`.
    neighbor_offsets : ndarray
        A one-dimensional array that contains the offsets to find the
        connected neighbors for any index in `image`.
    priority : ndarray, one-dimensional
        A one-dimensional array of indices indicating the order in which conflicting
        maxima are kept.
    minimal_distance : float
        The minimal euclidean distance allowed between non-conflicting maxima.
    """
    cdef:
        Py_ssize_t p, i, start_index, current_index, neighbor
        cnp.float64_t sq_distance,
        Py_ssize_t[::1] unravel_factors
        unsigned char[::1] queue_count,
        QueueWithHistory current_maximum, to_search, to_delete

    sq_distance = minimal_distance * minimal_distance
    queue_count = np.zeros(maxima.shape[0], dtype=np.uint8)

    # Calculate factors to unravel indices for `maxima` and `flags`:
    # -> omit the first dimension
    # -> multiply dimension with all following dimensions
    unravel_factors = np.array(shape[1:], dtype=np.intp)
    for i in range(unravel_factors.shape[0] - 2, -1, -1):
        unravel_factors[i] *= unravel_factors[i + 1]

    with nogil:
        queue_init(&current_maximum, 64)
        queue_init(&to_search, 64)
        queue_init(&to_delete, 64)

        try:
            for p in range(priority.shape[0]):
                start_index = priority[p]

                # Index was queued & dealt with earlier can be safely skipped
                if queue_count[start_index] == 1:
                    continue
                if maxima[start_index] == 0:
                    # This should never be the case and hints either at faulty values in
                    # `priority` or a bug in this algorithm
                    with gil:
                        raise ValueError(
                            "value {} in priority does not point to maxima"
                            .format(start_index)
                        )

                # Empty buffers of all queues
                queue_clear(&current_maximum)
                queue_clear(&to_search)
                queue_clear(&to_delete)

                # Find all points of the current maximum and queue in `current_maximum`,
                # queue the points surrounding the maximum in `to_search`
                queue_push(&current_maximum, &start_index)
                queue_count[start_index] = 1
                while queue_pop(&current_maximum, &current_index):
                    for i in range(neighbor_offsets.shape[0]):
                        neighbor = current_index + neighbor_offsets[i]
                        if not 0 <= neighbor < maxima.shape[0]:
                            continue
                        if queue_count[neighbor] == 1:
                            continue
                        if maxima[neighbor] == 1:
                            queue_push(&current_maximum, &neighbor)
                            queue_count[neighbor] = 1
                        else:
                            queue_push(&to_search, &neighbor)
                            queue_count[neighbor] = 1

                # Evaluate the space within the minimal distance of the current maximum
                while queue_pop(&to_search, &current_index):
                    # Check if `current_index` is in range of any point
                    # in `current_maximum`
                    queue_restore(&current_maximum)
                    while queue_pop(&current_maximum, &i):
                        if _sq_euclidean_distance(
                            current_index, i, unravel_factors
                        ) <= sq_distance:
                            # At least one point is close enough -> break early
                            break
                    else:
                        # Didn't find any point close enough
                        # -> we aren't in range anymore and can ignore this point
                        continue

                    # If another maximum is at `current_index`, queue it in `to_delete`
                    if maxima[current_index] == 1:
                        queue_push(&to_delete, &current_index)
                        # Set flag to 2, to indicate that it was queued twice:
                        # in `to_search` and `to_delete`
                        queue_count[current_index] = 2

                    # Queue neighbors of `current_index` for searching
                    for i in range(neighbor_offsets.shape[0]):
                        neighbor = current_index + neighbor_offsets[i]
                        if not 0 <= neighbor < maxima.shape[0]:
                            continue
                        if queue_count[neighbor] == 0:
                            queue_push(&to_search, &neighbor)
                            queue_count[neighbor] = 1

                # Restore empty range surrounding the current_maximum
                queue_restore(&to_search)
                while queue_pop(&to_search, &current_index):
                    # Decrease queue count to honor points which are queued a second
                    # time in `to_delete`
                    queue_count[current_index] -= 1

                # Remove maxima that are to close
                while queue_pop(&to_delete, &current_index):
                    maxima[current_index] = 0
                    # Find connected points of current deleted maximum
                    for i in range(neighbor_offsets.shape[0]):
                        neighbor = current_index + neighbor_offsets[i]
                        if not 0 <= neighbor < maxima.shape[0]:
                            continue
                        if queue_count[neighbor] == 0 and maxima[neighbor] == 1:
                            queue_push(&to_delete, &neighbor)
                            queue_count[neighbor] = 1

        finally:
            queue_exit(&current_maximum)
            queue_exit(&to_search)
            queue_exit(&to_delete)
