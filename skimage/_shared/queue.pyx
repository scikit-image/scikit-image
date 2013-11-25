cimport cqueue
from .queue cimport Queue


cdef class Queue:

    def __cinit__(self):
        self._c_queue = cqueue.queue_new()
        if self._c_queue is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._c_queue is not NULL:
            cqueue.queue_free(self._c_queue)

    cdef void push_tail(self, void* value):
        if not cqueue.queue_push_tail(self._c_queue, value):
            raise MemoryError()

    cdef void push_head(self, void* value):
        if not cqueue.queue_push_head(self._c_queue, value):
            raise MemoryError()

    cdef void* peek_head(self):
        return cqueue.queue_peek_head(self._c_queue)

    cdef void* pop_head(self):
        return cqueue.queue_pop_head(self._c_queue)

    cdef void* peek_tail(self):
        return cqueue.queue_peek_tail(self._c_queue)

    cdef void* pop_tail(self):
        return cqueue.queue_pop_tail(self._c_queue)

    cdef bint is_empty(self):
        return cqueue.queue_is_empty(self._c_queue)
