#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

"""Cython code used in extrema.py."""

cimport numpy as cnp
cnp.import_array()


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
        should be marked with "NOT_MAXIMUM".
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
        evaluation.
    
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
        evaluation.
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
        evaluation.
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
