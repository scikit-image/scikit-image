from libc.stdlib cimport free, malloc, realloc
from libc.stdint cimport uintptr_t


cdef struct Heap:
    Py_ssize_t items
    Py_ssize_t space
    Heapitem *data
    Heapitem **ptrs

cdef inline Heap *heap_from_numpy2() nogil:
    cdef Py_ssize_t k
    cdef Heap *heap
    heap = <Heap *> malloc(sizeof (Heap))
    heap.items = 0
    heap.space = 1000
    heap.data = <Heapitem *> malloc(heap.space * sizeof(Heapitem))
    heap.ptrs = <Heapitem **> malloc(heap.space * sizeof(Heapitem *))
    for k in range(heap.space):
        heap.ptrs[k] = heap.data + k
    return heap

cdef inline void heap_done(Heap *heap) nogil:
   free(heap.data)
   free(heap.ptrs)
   free(heap)

cdef inline void swap(Py_ssize_t a, Py_ssize_t b, Heap *h) nogil:
    h.ptrs[a], h.ptrs[b] = h.ptrs[b], h.ptrs[a]


######################################################
# heappop - inlined
#
# pop an element off the heap, maintaining heap invariant
#
# Note: heap ordering is the same as python heapq, i.e., smallest first.
######################################################
cdef inline void heappop(Heap *heap, Heapitem *dest) nogil:

    cdef Py_ssize_t i, smallest, l, r # heap indices

    #
    # Start by copying the first element to the destination
    #
    dest[0] = heap.ptrs[0][0]
    heap.items -= 1

    # if the heap is now empty, we can return, no need to fix heap.
    if heap.items == 0:
        return

    #
    # Move the last element in the heap to the first.
    #
    swap(0, heap.items, heap)

    #
    # Restore the heap invariant.
    #
    i = 0
    smallest = i
    while True:
        # loop invariant here: smallest == i

        # find smallest of (i, l, r), and swap it to i's position if necessary
        l = i * 2 + 1 #__left(i)
        r = i * 2 + 2 #__right(i)
        if l < heap.items:
            if smaller(heap.ptrs[l], heap.ptrs[i]):
                smallest = l
            if r < heap.items and smaller(heap.ptrs[r], heap.ptrs[smallest]):
                smallest = r
        else:
            # this is unnecessary, but trims 0.04 out of 0.85 seconds...
            break
        # the element at i is smaller than either of its children, heap
        # invariant restored.
        if smallest == i:
                break
        # swap
        swap(i, smallest, heap)
        i = smallest

##################################################
# heappush - inlined
#
# push the element onto the heap, maintaining the heap invariant
#
# Note: heap ordering is the same as python heapq, i.e., smallest first.
##################################################
cdef inline int heappush(Heap *heap, Heapitem *new_elem) nogil except -1:

    cdef Py_ssize_t child = heap.items
    cdef Py_ssize_t parent
    cdef Py_ssize_t k
    cdef Heapitem *new_data
    cdef Heapitem **new_ptr
    cdef uintptr_t original_data_ptr

    # grow if necessary
    if heap.items == heap.space:
        heap.space = heap.space * 2

        # Original pointer to silence compiler warnings about use-after-free:
        original_data_ptr = <uintptr_t>heap.data
        new_data = <Heapitem *>realloc(<void *>heap.data,
                        <Py_ssize_t>(heap.space * sizeof(Heapitem)))
        if not new_data:
            with gil:
                raise MemoryError()

        # If necessary, correct all stored pointers:
        if new_data != heap.data:
            for k in range(heap.items):
                # Calculate new pointers, `uintptr_t` avoids compiler warnings.
                heap.ptrs[k] = <Heapitem *>(<uintptr_t>new_data + (
                        <uintptr_t>heap.ptrs[k] - original_data_ptr))
        heap.data = new_data

        new_ptrs = <Heapitem **>realloc(<void *>heap.ptrs,
                    <Py_ssize_t>(heap.space * sizeof(Heapitem *)))
        if not new_ptrs:
            with gil:
                raise MemoryError()
        heap.ptrs = new_ptrs

        # Initialize newly allocated pointer storage:
        for k in range(heap.items, heap.space):
            heap.ptrs[k] = new_data + k

    # insert new data at child
    heap.ptrs[child][0] = new_elem[0]
    heap.items += 1

    # restore heap invariant, all parents <= children
    while child > 0:
        parent = (child + 1) // 2 - 1 # __parent(i)

        if smaller(heap.ptrs[child], heap.ptrs[parent]):
            swap(parent, child, heap)
            child = parent
        else:
            break

    return 0
