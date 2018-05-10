"""FIFO queue whose items can be iterated multiple times.

.. warning:: Expects "QueueItem" to be defined before inclusion!

This queue can be operated like a class. The structure stores the state of the
instance and the contained functions act as instance methods. The important
distinction compared to normal queues is that popping an element doesn't remove
it internally. Thus unless the queue's internal buffer is explicitly cleared
(see `queue_clear`) popped items can be restored at anytime using
`queue_restore`.

Example
-------
::

    cdef RestorableQueue queue
    cdef QueueItem item

    queue_init(&queue, 5)
    try:
        for i in range(10):
            item = <QueueItem>i
            queue_push(&queue, &item)

        while queue_pop(&queue, &item):
            print(item)  # Prints 0 to 9

        queue_restore(&queue)

        while queue_pop(&queue, &item):
            print(item)  # Prints 0 to 9 again
    finally:
        queue_exit(&queue)
"""

from libc.stdlib cimport malloc, realloc, free


# Store state of queue
cdef struct RestorableQueue:
    QueueItem* _buffer_ptr
    Py_ssize_t _buffer_size  # Maximal number of elements the buffer can store
    Py_ssize_t _index_valid  # Index to most recently inserted item
    Py_ssize_t _index_consumed  # Index to most recently consumed item


cdef inline void queue_init(RestorableQueue* self, Py_ssize_t buffer_size) nogil:
    """Initialize the queue and its buffer size.
    
    The size is defined as the number of queue items to fit into the initial 
    buffer, thus its true memory size is `buffer_size * sizeof(QueueItem)`. Be
    sure to call `queue_exit` after calling this function to free the allocated
    memory!
    """
    self._buffer_ptr = <QueueItem*>malloc(buffer_size * sizeof(QueueItem))
    if not self._buffer_ptr:
        with gil:
            raise MemoryError("couldn't allocate buffer")
    self._buffer_size = buffer_size
    self._index_consumed = -1
    self._index_valid = -1


cdef inline void queue_push(RestorableQueue* self, QueueItem* item_ptr) nogil:
    """Enqueue a new item."""
    self._index_valid += 1
    if self._buffer_size <= self._index_valid:
        _queue_grow_buffer(self)
    self._buffer_ptr[self._index_valid] = item_ptr[0]


cdef inline unsigned char queue_pop(RestorableQueue* self,
                                    QueueItem* item_ptr) nogil:
    """If not empty pop an item and return 1 otherwise return 0.
    
    The item is still preserved in the internal buffer and can be restored with
    `queue_restore`. To truly clear the internal buffer use `queue_clear`.
    """
    if 0 <= self._index_consumed + 1 <= self._index_valid:
        self._index_consumed += 1
        item_ptr[0] = self._buffer_ptr[self._index_consumed]
        return 1
    return 0


cdef inline void queue_restore(RestorableQueue* self) nogil:
    """Restore all consumed items to the queue."""
    self._index_consumed = -1


cdef inline void queue_clear(RestorableQueue* self) nogil:#
    """Remove all consumable items.
    
    After this the old items can't be restored with `queue_restore`.
    """
    self._index_consumed = -1
    self._index_valid = -1


cdef inline void queue_exit(RestorableQueue* self) nogil:
    """Free the buffer of the queue.
    
    Don't use the queue after this command unless `queue_init` is called again.
    """
    free(self._buffer_ptr)


cdef inline void _queue_grow_buffer(RestorableQueue* self) nogil:
    """Double the memory used for the buffer."""
    cdef QueueItem* new_buffer
    self._buffer_size *= 2
    new_buffer_ptr = <QueueItem*>realloc(
        self._buffer_ptr,
        self._buffer_size * sizeof(QueueItem)
    )
    if not new_buffer_ptr:
        with gil:
            raise MemoryError("couldn't reallocate buffer")
    self._buffer_ptr = new_buffer_ptr
