cimport cqueue

cdef class Queue:

    cdef cqueue.Queue* _c_queue
    cdef void push_tail(self, void* value)
    cdef void push_head(self, void* value)
    cdef void* peek_head(self)
    cdef void* pop_head(self)
    cdef void* peek_tail(self)
    cdef void* pop_tail(self)
    cdef bint is_empty(self)

