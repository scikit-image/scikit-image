'''_cpmorphology2.pyx - support routines for cpmorphology in Cython

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "Python.h":
    ctypedef int Py_intptr_t

cdef extern from "numpy/arrayobject.h":
    ctypedef class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef Py_intptr_t *dimensions
        cdef Py_intptr_t *strides
    cdef void import_array()
    cdef int  PyArray_ITEMSIZE(np.ndarray)

import_array()

@cython.boundscheck(False)
def skeletonize_loop(np.ndarray[dtype=np.uint8_t, ndim=2, 
                                negative_indices=False, mode='c'] result,
                     np.ndarray[dtype=np.int32_t, ndim=1,
                                negative_indices=False, mode='c'] i,
                     np.ndarray[dtype=np.int32_t, ndim=1,
                                negative_indices=False, mode='c'] j,
                     np.ndarray[dtype=np.int32_t, ndim=1,
                                negative_indices=False, mode='c'] order,
                     np.ndarray[dtype=np.uint8_t, ndim=1,
                                negative_indices=False, mode='c'] table):
    '''Inner loop of skeletonize function

    result - on input, the image to be skeletonized, on output the skeletonized
             image
    i,j    - the coordinates of each foreground pixel in the image
    order  - the index of each pixel, in the order of processing
    table  - the 512-element lookup table of values after transformation

    The loop determines whether each pixel in the image can be removed without
    changing the Euler number of the image. The pixels are ordered by
    increasing distance from the background which means a point nearer to
    the quench-line of the brushfire will be evaluated later than a
    point closer to the edge.
    '''
    cdef:
        np.int32_t accumulator
        np.int32_t index,order_index
        np.int32_t ii,jj

    for index in range(order.shape[0]):
        accumulator = 16
        order_index = order[index]
        ii = i[order_index]
        jj = j[order_index]
        if ii > 0:
            if jj > 0 and result[ii-1,jj-1]:
                accumulator += 1
            if result[ii-1,jj]:
                accumulator += 2
            if jj < result.shape[1]-1 and result[ii-1,jj+1]:
                    accumulator += 4
            if jj > 0 and result[ii,jj-1]:
                accumulator += 8
            if jj < result.shape[1]-1 and result[ii,jj+1]:
                accumulator += 32
            if ii < result.shape[0]-1:
                if jj > 0 and result[ii+1,jj-1]:
                    accumulator += 64
                if result[ii+1,jj]:
                    accumulator += 128
                if jj < result.shape[1]-1 and result[ii+1,jj+1]:
                    accumulator += 256
            result[ii,jj] = table[accumulator]

@cython.boundscheck(False)
def table_lookup_index(np.ndarray[dtype=np.uint8_t, ndim=2,
                                  negative_indices=False, mode='c'] image):
    '''Return an index into a table per pixel of a binary image

    Take the sum of true neighborhood pixel values where the neighborhood
    looks like this:
     1   2   4
     8  16  32
    64 128 256

    This code could be replaced by a convolution with the kernel:
    256 128 64
     32  16  8
      4   2  1
    but this runs about twice as fast because of inlining and the
    hardwired kernel.
    '''
    cdef:
        np.ndarray[dtype=np.int32_t, ndim=2, 
                   negative_indices=False, mode='c'] indexer
        np.int32_t *p_indexer
        np.uint8_t *p_image
        np.int32_t i_stride
        np.int32_t i_shape
        np.int32_t j_shape
        np.int32_t i
        np.int32_t j
        np.int32_t offset

    i_shape   = image.shape[0]
    j_shape   = image.shape[1]
    indexer = np.zeros((i_shape,j_shape),np.int32)
    p_indexer = <np.int32_t *>indexer.data
    p_image   = <np.uint8_t *>image.data
    i_stride  = image.strides[0]
    assert i_shape >= 3 and j_shape >= 3, "Please use the slow method for arrays < 3x3"

    for i in range(1,i_shape-1):
        offset = i_stride*i+1
        for j in range(1,j_shape-1):
            if p_image[offset]:
                p_indexer[offset+i_stride+1] += 1
                p_indexer[offset+i_stride] += 2
                p_indexer[offset+i_stride-1] += 4
                p_indexer[offset+1] += 8
                p_indexer[offset] += 16
                p_indexer[offset-1] += 32
                p_indexer[offset-i_stride+1] += 64
                p_indexer[offset-i_stride] += 128
                p_indexer[offset-i_stride-1] += 256
            offset += 1
    #
    # Do the corner cases (literally)
    #
    if image[0,0]:
        indexer[0,0] += 16
        indexer[0,1] += 8
        indexer[1,0] += 2
        indexer[1,1] += 1

    if image[0,j_shape-1]:
        indexer[0,j_shape-2] += 32
        indexer[0,j_shape-1] += 16
        indexer[1,j_shape-2] += 4
        indexer[1,j_shape-1] += 2

    if image[i_shape-1,0]:
        indexer[i_shape-2,0] += 128
        indexer[i_shape-2,1] += 64
        indexer[i_shape-1,0] += 16
        indexer[i_shape-1,1] += 8

    if image[i_shape-1,j_shape-1]:
        indexer[i_shape-2,j_shape-2] += 256
        indexer[i_shape-2,j_shape-1] += 128
        indexer[i_shape-1,j_shape-2] += 32
        indexer[i_shape-1,j_shape-1] += 16
    #
    # Do the edges
    #
    for j in range(1,j_shape-1):
        if image[0,j]:
            indexer[0,j-1] += 32
            indexer[0,j]   += 16
            indexer[0,j+1] += 8
            indexer[1,j-1] += 4
            indexer[1,j]   += 2
            indexer[1,j+1] += 1
        if image[i_shape-1,j]:
            indexer[i_shape-2,j-1] += 256
            indexer[i_shape-2,j]   += 128
            indexer[i_shape-2,j+1] += 64
            indexer[i_shape-1,j-1] += 32
            indexer[i_shape-1,j]   += 16
            indexer[i_shape-1,j+1] += 8

    for i in range(1,i_shape-1):
        if image[i,0]:
            indexer[i-1,0] += 128
            indexer[i,0]   += 16
            indexer[i+1,0] += 2
            indexer[i-1,1] += 64
            indexer[i,1]   += 8
            indexer[i+1,1] += 1
        if image[i,j_shape-1]:
            indexer[i-1,j_shape-2] += 256
            indexer[i,j_shape-2]   += 32
            indexer[i+1,j_shape-2] += 4
            indexer[i-1,j_shape-1] += 128
            indexer[i,j_shape-1]   += 16
            indexer[i+1,j_shape-1] += 2
    return indexer

@cython.boundscheck(False)
def grey_reconstruction_loop(np.ndarray[dtype=np.uint32_t, ndim=1,
                                        negative_indices = False,
                                        mode = 'c'] avalues,
                             np.ndarray[dtype=np.int32_t, ndim=1,
                                        negative_indices = False,
                                        mode = 'c'] aprev,
                             np.ndarray[dtype=np.int32_t, ndim=1,
                                        negative_indices = False,
                                        mode = 'c'] anext,
                             np.ndarray[dtype=np.int32_t, ndim=1,
                                        negative_indices = False,
                                        mode = 'c'] astrides,
                             np.int32_t current,
                             int image_stride):
    '''The inner loop for grey_reconstruction'''
    cdef:
        np.int32_t neighbor
        np.uint32_t neighbor_value
        np.uint32_t current_value
        np.uint32_t mask_value
        np.int32_t link
        int i
        np.int32_t nprev
        np.int32_t nnext
        int nstrides = astrides.shape[0]
        np.uint32_t *values = <np.uint32_t *>(avalues.data)
        np.int32_t *prev = <np.int32_t *>(aprev.data)
        np.int32_t *next = <np.int32_t *>(anext.data)
        np.int32_t *strides = <np.int32_t *>(astrides.data)
    
    while current != -1:
        if current < image_stride:
            current_value = values[current]
            if current_value == 0:
                break
            for i in range(nstrides):
                neighbor = current + strides[i]
                neighbor_value = values[neighbor]
                # Only do neighbors less than the current value
                if neighbor_value < current_value:
                    mask_value = values[neighbor + image_stride]
                    # Only do neighbors less than the mask value
                    if neighbor_value < mask_value:
                        # Raise the neighbor to the mask value if
                        # the mask is less than current
                        if mask_value < current_value:
                            link = neighbor + image_stride
                            values[neighbor] = mask_value
                        else:
                            link = current
                            values[neighbor] = current_value
                        # unlink the neighbor
                        nprev = prev[neighbor]
                        nnext = next[neighbor]
                        next[nprev] = nnext
                        if nnext != -1:
                            prev[nnext] = nprev
                        # link the neighbor after the link
                        nnext = next[link]
                        next[neighbor] = nnext
                        prev[neighbor] = link
                        if nnext >= 0:
                            prev[nnext] = neighbor
                            next[link] = neighbor
        current = next[current]
        
@cython.boundscheck(False)
def _all_connected_components(np.ndarray[dtype=np.uint32_t, ndim=1,
                                         negative_indices = False,
                                         mode = 'c'] i_a,
                              np.ndarray[dtype=np.uint32_t, ndim=1,
                                         negative_indices = False,
                                         mode = 'c'] j_a,
                              np.ndarray[dtype=np.uint32_t, ndim=1,
                                         negative_indices = False,
                                         mode = 'c'] indexes_a,
                              np.ndarray[dtype=np.uint32_t, ndim=1,
                                         negative_indices = False,
                                         mode = 'c'] counts_a,
                              np.ndarray[dtype=np.uint32_t, ndim=1,
                                         negative_indices = False,
                                         mode = 'c'] label_a):
    '''Inner loop for the all_connected_components algorithm
    
    i,j - vectors giving the edges between vertices. These vectors must
          be pre-sorted by i then j. There should be one j,i edge for
          every i,j edge for all_connected_components.
          
    indexes - 1 index per vertex # into the i,j array giving the index
              of the first edge for vertex i
              
    counts - 1 count per vertex # of the # of edges for vertex i
    
    label - one array element per vertex, the label assigned to this
            vertex's connected components. On exit, this is the vertex
            number of the first vertex processed in the connected component.
    '''
    cdef:
        # n = # of vertices
        np.uint32_t n = counts_a.shape[0]
        np.ndarray[dtype=np.uint8_t, ndim=1,
                   negative_indices = False,
                   mode = 'c'] being_processed_a = np.zeros(n, np.uint8)
        np.ndarray[dtype=np.uint32_t, ndim=1,
                   negative_indices = False,
                   mode = 'c'] visit_index_a = np.zeros(n, np.uint32)
        np.ndarray[dtype=np.uint32_t, ndim=1,
                   negative_indices = False,
                   mode = 'c'] stack_v_a = np.zeros(n, np.uint32)
        np.ndarray[dtype=np.uint32_t, ndim=1,
                   negative_indices = False,
                   mode = 'c'] v_idx_a  = np.zeros(n, np.uint32)
        np.ndarray[dtype=np.uint32_t, ndim=1,
                   negative_indices = False,
                   mode = 'c'] stack_being_processed_a = np.zeros(n, np.uint32)
        #
        # This is the recursion depth for recursive calls to process the
        # connected components.
        #
        np.uint32_t  stack_ptr = 0
        #
        # cur_index gives the next visit_index to be assigned. The visit_index
        # orders vertices according to when they were first visited
        #
        np.uint32_t  cur_index = 0
        #
        # raw pointer to the "from" vertices in the edge list (unused)
        #
        np.uint32_t *i = <np.uint32_t *>(i_a.data)
        #
        # raw pointer to the "to" vertices in the edge list
        #
        np.uint32_t *j = <np.uint32_t *>(j_a.data)
        #
        # raw pointer to the index into j for each vertex
        #
        np.uint32_t *indexes = <np.uint32_t *>(indexes_a.data)
        #
        # raw pointer to the # of edges per vertex
        #
        np.uint32_t *counts = <np.uint32_t *>(counts_a.data)
        #
        # the label per vertex
        #
        np.uint32_t *label = <np.uint32_t *>(label_a.data)
        #
        # The recursive "function" has a single argument - the vertex to be
        # processed, so this mimics a stack where v is pushed on entry
        # and popped on exit.
        #
        np.uint32_t *stack_v = <np.uint32_t *>(stack_v_a.data)
        #
        # The recursive function has a single local variable
        # v_idx which is the index of the current edge being processed.
        # On entry, this is UNDEFINED to indicate that the recursive function
        # should initialize "v". Afterwards, it cycles from 0 to the count.
        # When it reaches the count, the recursive function "exits"
        #
        np.uint32_t *v_idx = <np.uint32_t *>(v_idx_a.data)
        #
        # Holds the index of the top-level v in the top-level loop
        #
        np.uint32_t v
        #
        # Holds the v inside the recursive function
        #
        np.uint32_t vv
        #
        # Holds the vertex at the opposite side of the edge from vv
        #
        np.uint32_t v1
        #
        # A value to indicate something that's not been processed
        #
        np.uint32_t UNDEFINED = -1
        
    label_a[:] = UNDEFINED
    v_idx_a[:] = UNDEFINED
        
    for v in range(n):
        if label[v] == UNDEFINED:
            stack_v[0] = v
            stack_ptr = 1
            while(stack_ptr > 0):
                vv = stack_v[stack_ptr-1]
                if v_idx[vv] == UNDEFINED:
                    #
                    # Start of recursive function for a vertex: set the vertex
                    #     label.
                    #
                    label[vv] = cur_index
                    v_idx[vv] = 0
                if v_idx[vv] < counts[vv]:
                    # For each edge of v, push other vertex on stack 
                    # if unprocessed.
                    #
                    v1 = j[indexes[vv] + v_idx[vv]]
                    v_idx[vv] += 1
                    if label[v1] == UNDEFINED:
                        stack_v[stack_ptr] = v1
                        stack_ptr += 1
                else:
                    #
                    # We have processed every edge for vv.
                    #
                    stack_ptr -= 1
            cur_index += 1                                    

@cython.boundscheck(False)
def index_lookup(np.ndarray[dtype=np.int32_t, ndim=1, negative_indices=False] index_i, 
                 np.ndarray[dtype=np.int32_t, ndim=1, negative_indices=False] index_j, 
                 np.ndarray[dtype=np.uint32_t, ndim=2, negative_indices=False] image,
                 table_in, 
                 iterations=None):
    '''Perform a table lookup for only the indexed pixels
    
    For morphological operations that only convert 1 to 0, the set of
    resulting pixels is always a subset of the input set. Therefore, when
    repeating, it will be faster to operate only on the subsets especially
    when the results are 1-d or 0-d objects.
    
    This function returns a new index_i and index_j array of the pixels
    that survive the operation. The image is modified in-place to remove
    the pixels that did not survive.
    
    index_i - an array of row indexes into the image.
    index_j - a similarly-shaped array of column indexes.
    image - the binary image: *NOTE* add a row and column of border values
            to the original image to account for pixels on the edge of the
            image.
    iterations - # of iterations to do, default is "forever"
    
    The idea of index_lookup was taken from
    http://blogs.mathworks.com/steve/2008/06/13/performance-optimization-for-applylut/
    which, apparently, is how Matlab achieved its bwmorph speedup.
    '''
    cdef:
        np.ndarray[dtype=np.uint8_t, ndim=1, negative_indices=False] table = table_in.astype(np.uint8)
        np.uint32_t center, hit_count, idx, indexer
        np.int32_t idxi, idxj

    if iterations == None:
        # Worst case - remove one per iteration
        iterations = len(index_i)

    for i in range(iterations):
        hit_count = len(index_i)
        with nogil:
            #
            # For integer images (i.e., labels), a neighbor point is
            # "background" if it doesn't match the central value. This
            # lets adjacent labeled objects shrink independently of each
            # other.
            #
            for 0 <= idx < hit_count:
                idxi, idxj = index_i[idx], index_j[idx]
                center = image[idxi, idxj]
                indexer =          ((image[idxi - 1, idxj - 1] == center) * 1 +
                                    (image[idxi - 1, idxj]     == center) * 2 +
                                    (image[idxi - 1, idxj + 1] == center) * 4 +
                                    (image[idxi,     idxj - 1] == center) * 8 +
                                    16 +
                                    (image[idxi,     idxj + 1] == center) * 32 +
                                    (image[idxi + 1, idxj - 1] == center) * 64 +
                                    (image[idxi + 1, idxj]     == center) * 128 +
                                    (image[idxi + 1, idxj + 1] == center) * 256)
                if table[indexer] == 0:
                    # mark for deletion
                    index_i[idx] = -index_i[idx]

            # remove marked pixels
            for 0 <= idx < hit_count:
                idxi, idxj = index_i[idx], index_j[idx]
                if idxi < 0:
                    image[-idxi, idxj] = 0

        index_j = index_j[index_i >= 0]
        index_i = index_i[index_i >= 0]
        if len(index_i) == hit_count:
            break

    return (index_i, index_j)

def prepare_for_index_lookup(image, border_value):
    '''Return the index arrays of "1" pixels and an image with an added border
    
    The routine, index_lookup takes an array of i indexes, an array of
    j indexes and an image guaranteed to be indexed successfully by 
    index_<i,j>[:] +/- 1. This routine constructs an image with added border
    pixels... evilly, the index, 0 - 1, lands on the border because of Python's
    negative indexing convention.
    '''
    if np.issubdtype(image.dtype, float):
        image = image.astype(bool)
    image_i, image_j = np.argwhere(image.astype(bool)).transpose().astype(np.int32) + 1
    output_image = (np.ones(np.array(image.shape)+2,image.dtype) if border_value
                    else np.zeros(np.array(image.shape)+2, image.dtype))
    output_image[1:image.shape[0]+1, 1:image.shape[1]+1] = image
    return (image_i, image_j, output_image.astype(np.uint32))


def extract_from_image_lookup(orig_image, index_i, index_j):
    output = np.zeros(orig_image.shape, orig_image.dtype)
    output[index_i - 1, index_j - 1] = orig_image[index_i - 1, index_j - 1]
    return output

def ptrsize():
    '''The number of bytes in a pointer'''
    return sizeof(int *)

@cython.boundscheck(False)
def fill_labeled_holes_loop(
    np.ndarray[dtype = np.uint32_t, ndim = 1, negative_indices = False] i,
    np.ndarray[dtype = np.uint32_t, ndim = 1, negative_indices = False] j,
    np.ndarray[dtype = np.uint32_t, ndim = 1, negative_indices = False] idx,
    np.ndarray[dtype = np.uint32_t, ndim = 1, negative_indices = False] i_count,
    np.ndarray[dtype = np.uint8_t,  ndim = 1, negative_indices = False] is_not_hole,
    np.ndarray[dtype = np.uint32_t, ndim = 1, negative_indices = False] adjacent_non_hole,
    np.ndarray[dtype = np.uint32_t, ndim = 1, negative_indices = False] to_do,
    int lcount,
    int to_do_count):
    '''Run the graph loop portions of fill_labeled_holes
    
    i, j - the labels of pairs of touching objects
    idx - the index to the j for a particular i
    i_counts - the number of j for a particular i
    is_not_hole - on entry, a boolean array with one element per label. On exit,
                  True if the label is not a hole to be filled.
    adjacent_non_hole - on entry, an integer array set to zero with one
                        element per label. On exit, for each hole, the label
                        that should be used to fill it.
    to_do - one element per label. The initial non-hole labels (with to_do_count
            as the number on the list).
    lcount - all labels with a label # of lcount or below are objects, all above
             are background.
    '''
    cdef:
        int ii,jj,jidx
        int n = len(is_not_hole)
        int *p_idx = <int *>(idx.data)
        int *p_i_count = <int *>(i_count.data)
        int *p_to_do = <int *>(to_do.data)
        char *p_is_not_hole = <char *>(is_not_hole.data)
        int *p_adjacent_non_hole = <int *>(adjacent_non_hole.data)
        int *p_i = <int *>(i.data)
        int *p_j = <int *>(j.data)
    with nogil:
        while to_do_count > 0:
            ii = p_to_do[to_do_count - 1]
            to_do_count -= 1
            #
            # jj are the adjacencies to ii
            #
            for jidx from 0 <= jidx < p_i_count[ii]:
                jj = p_j[p_idx[ii] + jidx]
                if p_is_not_hole[jj] == 0:
                    if ii <= lcount:
                        #
                        # i labels an object. Label any object that is adjacent to
                        # a different object.
                        #
                        if p_adjacent_non_hole[jj] == 0:
                            p_adjacent_non_hole[jj] = ii
                            continue
                        elif p_adjacent_non_hole[jj] == ii:
                            continue
                    elif jj > lcount:
                        #
                        # i labels background. Label any foreground object touching it.
                        #
                        continue
                    p_is_not_hole[jj] = 1
                    p_to_do[to_do_count] = jj
                    to_do_count += 1
        #
        # Now we need to walk the graph of hole objects. There are holes
        # that are touching non-holes. These are the ones where adjacent_non_hole
        # is not zero. There are holes that are inside holes (objects inside holes)
        # and holes that are inside holes that are inside holes (holes inside
        # objects inside holes) and so on until you get to the bullseye of some
        # mega-archery-target.
        #
        to_do_count = 0
        for jj from 0 <= jj < n:
            if p_is_not_hole[jj] != 0 and p_adjacent_non_hole[jj] != 0:
                p_to_do[to_do_count] = jj
                to_do_count += 1
        while to_do_count > 0:
            ii = p_to_do[to_do_count - 1]
            to_do_count -= 1
            for jidx from 0 <= jidx < p_i_count[ii]:
                jj = p_j[p_idx[ii] + jidx]
                if p_adjacent_non_hole[jj] == 0:
                    p_adjacent_non_hole[jj] = p_adjacent_non_hole[ii]
                    p_to_do[to_do_count] = jj
                    to_do_count += 1
    