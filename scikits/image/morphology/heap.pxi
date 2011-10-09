"""
Originally part of CellProfiler, code licensed under both GPL and BSD
licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
"""
import numpy as np
cimport numpy as np
cimport cython
from sys import stderr

cdef extern from "stdlib.h":
   ctypedef unsigned long size_t
   void free(void *ptr)
   void *malloc(size_t size)
   void *realloc(void *ptr, size_t size) nogil

cdef struct Heap:
    unsigned int items
    unsigned int width
    unsigned int space
    np.int32_t *data
    np.int32_t **ptrs


cdef inline Heap *heap_from_numpy2(object np_heap):
    cdef unsigned int k
    cdef Heap *heap 
    heap = <Heap *> malloc(sizeof (Heap))
    heap.items = np_heap.shape[0]
    heap.width = np_heap.shape[1]
    heap.space = max(heap.items, 1000)
    heap.data = <np.int32_t *> malloc(heap.space * heap.width * sizeof(np.int32_t))
    heap.ptrs = <np.int32_t **> malloc(heap.space * sizeof(np.int32_t *))
    tmp = np_heap.astype(np.int32).flatten('C')
    for k from 0 <= k < heap.items * heap.width:
        heap.data[k] = <np.int32_t> tmp[k]
    for k from 0 <= k < heap.space:
        heap.ptrs[k] = heap.data + k * heap.width
    return heap

cdef inline void heap_done(Heap *heap):
   free(heap.data)
   free(heap.ptrs)
   free(heap)

cdef inline int smaller(unsigned int a, unsigned int b, Heap *h) nogil:
    cdef unsigned int k
    cdef np.int32_t *ap = h.ptrs[a]
    cdef np.int32_t *bp = h.ptrs[b]
    if ap[0] == bp[0]:
        for k from 1 <= k < h.width:
            if ap[k] == bp[k]:
                continue
            if ap[k] < bp[k]:
                return 1
            break
        return 0
    elif ap[0] < bp[0]:
       return 1
    return 0

cdef inline void swap(unsigned int a, unsigned int b, Heap *h) nogil:
    h.ptrs[a], h.ptrs[b] = h.ptrs[b], h.ptrs[a]


######################################################
# heappop - inlined
#
# pop an element off the heap, maintaining heap invariant
# 
# Note: heap ordering is the same as python heapq, i.e., smallest first.
######################################################
cdef inline void heappop(Heap *heap,
                  np.int32_t *dest) nogil:
    cdef unsigned int i, smallest, l, r # heap indices
    cdef unsigned int k
    
    #
    # Start by copying the first element to the destination
    #
    for k from 0 <= k < heap.width:
        dest[k] = heap.ptrs[0][k]
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
        l = i*2+1 #__left(i)
        r = i*2+2 #__right(i)
        if l < heap.items:
            if smaller(l, i, heap):
                smallest = l
            if r < heap.items and smaller(r, smallest, heap):
                smallest = r
        else:
            # this is unnecessary, but trims 0.04 out of 0.85 seconds...
            break
        # the element at i is smaller than either of its children, heap invariant restored.
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
cdef inline void heappush(Heap *heap,
                          np.int32_t *new_elem) nogil:
  cdef unsigned int child         = heap.items
  cdef unsigned int parent
  cdef unsigned int k
  cdef np.int32_t *new_data

  # grow if necessary
  if heap.items == heap.space:
      heap.space = heap.space * 2
      new_data = <np.int32_t *> realloc(<void *> heap.data, <size_t> (heap.space * heap.width * sizeof(np.int32_t)))
      heap.ptrs = <np.int32_t **> realloc(<void *> heap.ptrs, <size_t> (heap.space * sizeof(np.int32_t *)))
      for k from 0 <= k < heap.items:
          heap.ptrs[k] = new_data + (heap.ptrs[k] - heap.data)
      for k from heap.items <= k <  heap.space:
          heap.ptrs[k] = new_data + k * heap.width
      heap.data = new_data

  # insert new data at child
  
  for k from 0 <= k < heap.width:
      heap.ptrs[child][k] = new_elem[k]
  heap.items += 1

  # restore heap invariant, all parents <= children
  while child>0:
      parent = (child + 1) / 2 - 1 # __parent(i)
      
      if smaller(child, parent, heap):
          swap(parent, child, heap)
          child = parent
      else:
          break
