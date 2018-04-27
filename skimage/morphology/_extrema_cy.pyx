#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.stdlib cimport malloc, realloc, free


ctypedef fused dtype_t:
    cnp.uint8_t
    cnp.uint16_t
    cnp.double_t


ctypedef Py_ssize_t QueueItem


cdef:
    struct Queue:
        # A queue whose items can be restored after consumption.
        QueueItem* _buffer_ptr
        Py_ssize_t _buffer_size, _index_valid, _index_consumed

    void q_init(Queue* self, Py_ssize_t buffer_size) nogil:
        """Initialize the queue and its buffer."""
        self._buffer_ptr = <QueueItem*>malloc(buffer_size * sizeof(QueueItem))
        if not self._buffer_ptr:
            with gil:
                raise MemoryError("couldn't allocate buffer")
        self._buffer_size = buffer_size
        self._index_consumed = -1
        self._index_valid = -1

    void q_free_buffer(Queue* self) nogil:
        """Free the buffer of the queue."""
        free(self._buffer_ptr)

    inline void q_restore(Queue* self) nogil:
        """Restore all consumed items to the queue."""
        self._index_consumed = -1

    inline void q_clear(Queue* self) nogil:#
        """Remove all items in the queue."""
        self._index_consumed = -1
        self._index_valid = -1

    inline void q_push(Queue* self, QueueItem* item_ptr) nogil:
        """Enqueue a new item."""
        self._index_valid += 1
        if self._buffer_size <= self._index_valid:
            _q_grow_buffer(self)
        self._buffer_ptr[self._index_valid] = item_ptr[0]

    inline unsigned char q_pop(Queue* self, QueueItem* item_ptr) nogil:
        """Dequeue / consume an item."""
        if 0 <= self._index_consumed + 1 <= self._index_valid:
            self._index_consumed += 1
            item_ptr[0] = self._buffer_ptr[self._index_consumed]
            return 1
        return 0

    void _q_grow_buffer(Queue* self) nogil:
        """Double the memory used for the buffer."""
        cdef QueueItem* new_buffer

        # TODO prevent integer overflow!
        self._buffer_size *= 2
        new_buffer_ptr = <QueueItem*>realloc(
            self._buffer_ptr,
            self._buffer_size * sizeof(QueueItem)
        )
        if not new_buffer_ptr:
            with gil:
                raise MemoryError("couldn't reallocate buffer")
        self._buffer_ptr = new_buffer_ptr


def _local_maxima(dtype_t[::1] image not None,
                  unsigned char[::1] flags,
                  Py_ssize_t[::1] neighbour_offsets not None):
    """Detect local maxima in n-dimensional array.

    Parameters
    ----------
    image : ndarray, one-dimensional
        The raveled view of a n-dimensional array.
    flags : ndarray
        An array of flags that is used to store the state of each pixel during
        evaluation.
    neighbour_offsets : ndarray
        A one-dimensional array that contains the offsets to find the
        connected neighbours for any index in `image`.

    Returns
    -------
    is_maximum : ndarray
        A "boolean" arrray that is 1 where local maxima where exist.
    """
    cdef:
        Queue queue
        Py_ssize_t i, i_max, i_ahead

    # Current flag meanings:
    # 3 - first or last index in a dimension
    # 2 - potentially part of a maximum
    # 1 - not used in first loop
    # 0 - not evaluated or none of the above is true

    with nogil:
        i = 1
        i_max = image.shape[0]
        while i < i_max:
            if image[i - 1] < image[i] and flags[i] != 3:
                # Potential maximum (in last dimension) is found, find other
                # edge of current plateau or "edge of dimension"
                i_ahead = i + 1
                while image[i] == image[i_ahead] and flags[i_ahead] != 3:
                    i_ahead += 1
                if image[i] > image[i_ahead]:
                    # Found local maximum (in one dimension), mark all parts of
                    # the plateau as potential maximum
                    flags[i:i_ahead] = 2
                i = i_ahead
            else:
                i += 1

        # Initialize a buffer used to queue positions while evaluating each
        # potential maximum (flagged with 2)
        q_init(&queue, 64)
        try:
            for i in range(image.shape[0]):
                if flags[i] == 2:
                    # Index is potentially part of a maximum:
                    # Find all samples part of the plateau and fill with 0
                    # or 1 depending on whether it's a true maximum
                    _fill_plateau(image, flags, neighbour_offsets, &queue, i)
        finally:
            q_free_buffer(&queue)


cdef inline void _fill_plateau(
        dtype_t[::1] image, unsigned char[::1] flags,
        Py_ssize_t[::1] neighbour_offsets, Queue* queue_ptr,
        Py_ssize_t start_index) nogil:
    """Fill with 1 if plateau is local maximum else with 0."""
    cdef:
        dtype_t h
        unsigned char true_maximum
        QueueItem current_index, neighbor

    h = image[start_index]
    true_maximum = 1 # Boolean flag

    # Current / new flag meanings:
    # 3 - first or last value in a dimension
    # 2 - not used here
    # 1 - index was queued and might still be part of maximum
    # 0 - none of the above is true
    # Therefore mark current index as true maximum as an initial guess
    flags[start_index] = 1

    # And queue start position after clearing the buffer
    q_clear(queue_ptr)
    q_push(queue_ptr, &start_index)

    # Break loop if all queued positions were evaluated
    while q_pop(queue_ptr, &current_index):
        # Look at all neighbouring samples
        for i in range(neighbour_offsets.shape[0]):
            neighbor = current_index + neighbour_offsets[i]
            if image[neighbor] == h:
                # Value is part of plateau
                if flags[neighbor] == 3:
                    # Plateau touches border and can't be maximum
                    true_maximum = 0
                elif flags[neighbor] != 1:
                    # Index wasn't queued already, do so now
                    q_push(queue_ptr, &neighbor)
                    flags[neighbor] = 1
            elif image[neighbor] > h:
                # Current plateau can't be maximum because it borders a
                # larger one
                true_maximum = 0

    if not true_maximum:
        q_restore(queue_ptr)
        # Initial guess was wrong -> replace 1 with 0 for plateau
        while q_pop(queue_ptr, &neighbor):
            flags[neighbor] = 0
