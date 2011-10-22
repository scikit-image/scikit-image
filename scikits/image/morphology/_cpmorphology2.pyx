'''_cpmorphology2.pyx - support routines for cpmorphology in Cython

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
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
            if jj > 0 and result[ii - 1, jj - 1]:
                accumulator += 1
            if result[ii - 1, jj]:
                accumulator += 2
            if jj < result.shape[1] - 1 and result[ii - 1, jj + 1]:
                    accumulator += 4
            if jj > 0 and result[ii, jj - 1]:
                accumulator += 8
            if jj < result.shape[1] - 1 and result[ii, jj + 1]:
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
    """
    Return an index into a table per pixel of a binary image

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
    """
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
    indexer = np.zeros((i_shape, j_shape), np.int32)
    p_indexer = <np.int32_t *>indexer.data
    p_image   = <np.uint8_t *>image.data
    i_stride  = image.strides[0]
    assert i_shape >= 3 and j_shape >= 3, \
        "Please use the slow method for arrays < 3x3"

    for i in range(1, i_shape-1):
        offset = i_stride* i + 1
        for j in range(1, j_shape - 1):
            if p_image[offset]:
                p_indexer[offset + i_stride + 1] += 1
                p_indexer[offset + i_stride] += 2
                p_indexer[offset + i_stride - 1] += 4
                p_indexer[offset + 1] += 8
                p_indexer[offset] += 16
                p_indexer[offset - 1] += 32
                p_indexer[offset - i_stride + 1] += 64
                p_indexer[offset - i_stride] += 128
                p_indexer[offset - i_stride - 1] += 256
            offset += 1
    #
    # Do the corner cases (literally)
    #
    if image[0, 0]:
        indexer[0, 0] += 16
        indexer[0, 1] += 8
        indexer[1, 0] += 2
        indexer[1, 1] += 1

    if image[0, j_shape - 1]:
        indexer[0, j_shape - 2] += 32
        indexer[0, j_shape - 1] += 16
        indexer[1, j_shape - 2] += 4
        indexer[1, j_shape - 1] += 2

    if image[i_shape - 1, 0]:
        indexer[i_shape - 2, 0] += 128
        indexer[i_shape - 2, 1] += 64
        indexer[i_shape - 1, 0] += 16
        indexer[i_shape - 1, 1] += 8

    if image[i_shape - 1, j_shape - 1]:
        indexer[i_shape - 2, j_shape - 2] += 256
        indexer[i_shape - 2, j_shape - 1] += 128
        indexer[i_shape - 1, j_shape - 2] += 32
        indexer[i_shape - 1, j_shape - 1] += 16
    #
    # Do the edges
    #
    for j in range(1, j_shape - 1):
        if image[0, j]:
            indexer[0, j - 1] += 32
            indexer[0, j]   += 16
            indexer[0, j + 1] += 8
            indexer[1, j - 1] += 4
            indexer[1, j]   += 2
            indexer[1, j + 1] += 1
        if image[i_shape - 1, j]:
            indexer[i_shape - 2, j - 1] += 256
            indexer[i_shape - 2, j]   += 128
            indexer[i_shape - 2, j + 1] += 64
            indexer[i_shape - 1, j - 1] += 32
            indexer[i_shape - 1, j]   += 16
            indexer[i_shape - 1, j + 1] += 8

    for i in range(1, i_shape - 1):
        if image[i, 0]:
            indexer[i - 1, 0] += 128
            indexer[i, 0]   += 16
            indexer[i + 1, 0] += 2
            indexer[i - 1, 1] += 64
            indexer[i, 1]   += 8
            indexer[i + 1, 1] += 1
        if image[i, j_shape - 1]:
            indexer[i - 1, j_shape - 2] += 256
            indexer[i, j_shape - 2]   += 32
            indexer[i + 1, j_shape - 2] += 4
            indexer[i - 1, j_shape - 1] += 128
            indexer[i, j_shape - 1]   += 16
            indexer[i + 1, j_shape - 1] += 2
    return indexer

@cython.boundscheck(False)
def index_lookup(np.ndarray[dtype=np.int32_t, ndim=1, 
                                negative_indices=False] index_i, 
                 np.ndarray[dtype=np.int32_t, ndim=1, 
                                negative_indices=False] index_j, 
                 np.ndarray[dtype=np.uint32_t, ndim=2, 
                                negative_indices=False] image,
                 table_in, 
                 iterations=None):
    """
    Perform a table lookup for only the indexed pixels
    
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
    """
    cdef:
        np.ndarray[dtype=np.uint8_t, ndim=1, 
            negative_indices=False] table = table_in.astype(np.uint8)
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
                indexer =   ((image[idxi - 1, idxj - 1] == center) * 1 +
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
    """
    Return the index arrays of "1" pixels and an image with an added border
    
    The routine, index_lookup takes an array of i indexes, an array of
    j indexes and an image guaranteed to be indexed successfully by 
    index_<i,j>[:] +/- 1. This routine constructs an image with added border
    pixels... evilly, the index, 0 - 1, lands on the border because of Python's
    negative indexing convention.
    """
    if np.issubdtype(image.dtype, float):
        image = image.astype(bool)
    image_i, image_j = np.argwhere(image.astype(bool)).transpose().\
                                    astype(np.int32) + 1
    output_image = (np.ones(np.array(image.shape) + 2, image.dtype) \
                    if border_value
                    else np.zeros(np.array(image.shape) + 2, image.dtype))
    output_image[1:image.shape[0] + 1, 1:image.shape[1] + 1] = image
    return (image_i, image_j, output_image.astype(np.uint32))


def extract_from_image_lookup(orig_image, index_i, index_j):
    """
    Extract only one pixel
    """
    output = np.zeros(orig_image.shape, orig_image.dtype)
    output[index_i - 1, index_j - 1] = orig_image[index_i - 1, index_j - 1]
    return output

