"""_criteria.pyx - flooding algorithm for criteria based openings and closings

The implementation is inspired by the implementation of the watershed
transformation (watershed.pyx) and is also based on the implementation
of the hierarchical queue provided in heap_general.pyx
"""
#import numpy as np
#from IPython.html.widgets.interaction import _get_min_max_value
#from ._watershed import _euclid_dist

import numpy as np
from libc.math cimport sqrt

from libc.stdlib cimport free, malloc, realloc

cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_FLOAT64_t
ctypedef np.int32_t DTYPE_INT32_t
ctypedef np.uint32_t DTYPE_UINT32_t
ctypedef np.uint64_t DTYPE_UINT64_t
ctypedef np.int64_t DTYPE_INT64_t
ctypedef np.uint8_t DTYPE_BOOL_t


#from criteria_classes import _Area

#from numpy cimport uint8_t, uint16_t, double_t


ctypedef fused dtype_t:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t


include "heap_watershed.pxi"
include "util.pxi"

cdef struct Area:
    DTYPE_UINT64_t area
    DTYPE_UINT64_t equivalent_label
    DTYPE_FLOAT64_t stop_level
    DTYPE_FLOAT64_t val_of_min
#    DTYPE_FLOAT64_t volume

cdef void _update_area(Area* p_element):
    p_element.area += 1

@cython.boundscheck(False)
def _criteria_closing(dtype_t[::1] image, #DTYPE_FLOAT64_t[::1] image,
                 DTYPE_UINT64_t area_threshold,
                 DTYPE_UINT64_t[::1] label_img,
                 DTYPE_INT32_t[::1] structure,
                 DTYPE_BOOL_t[::1] mask,
                 np.int32_t[::1] strides,
                 DTYPE_FLOAT64_t eps,
                 np.double_t compactness,
                 dtype_t[::1] output, #DTYPE_FLOAT64_t[::1] output,
                 void update_functor,
                 ):
    """Perform criteria based closings using.

    Parameters
    ----------

    image : array of float
        The flattened image pixels.
    marker_locations : array of int
        The raveled coordinates of the initial markers (aka seeds) for the
        watershed. NOTE: these should *all* point to nonzero entries in the
        output, or the algorithm will never terminate and blow up your memory!
    structure : array of int
        A list of coordinate offsets to compute the raveled coordinates of each
        neighbor from the raveled coordinates of the current pixel.
    mask : array of int
        An array of the same shape as `image` where each pixel contains a
        nonzero value if it is to be considered for flooding with watershed,
        zero otherwise. NOTE: it is *essential* that the border pixels (those
        with neighbors falling outside the volume) are all set to zero, or
        segfaults could occur.
    strides : array of int
        An array representing the number of steps to move along each dimension.
        This is used in computing the Euclidean distance between raveled
        indices.
    compactness : float
        A value greater than 0 implements the compact watershed algorithm
        (see .py file).
    output : array of int
        The output array, which must already contain nonzero entries at all the
        seed locations.
    wsl : bool
        Parameter indicating whether the watershed line is calculated.
        If wsl is set to True, the watershed line is calculated.
    """
    cdef Heapitem elem
    cdef Heapitem new_elem

    cdef Py_ssize_t nneighbors = structure.shape[0]
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t age = 1
    cdef Py_ssize_t index = 0
    cdef Py_ssize_t lab_index = 0

    cdef DTYPE_UINT64_t l = 0
    cdef DTYPE_UINT64_t equ_l = 0
    cdef DTYPE_UINT64_t label1 = 0
    cdef DTYPE_UINT64_t label2 = 0

    cdef DTYPE_FLOAT64_t min_val, max_val;
    cdef DTYPE_FLOAT64_t value;
    cdef DTYPE_FLOAT64_t level_before;

    # hierarchical queue for the flooding
    cdef Heap *hp = <Heap *> heap_from_numpy2()

    # array for characterization of minima
    cdef int number_of_minima = np.max(label_img) + 1
    cdef Area* area_vec = <Area *>malloc(sizeof(Area) * number_of_minima)

    # type dependent arrays (cannot be part of the structure due to cython limitation)
    #cdef dtype_t *stop_level = <dtype_t *>malloc(sizeof(dtype_t) * number_of_minima)
    #cdef dtype_t *value_of_min = <dtype_t *>malloc(sizeof(dtype_t) * number_of_minima)

    min_val = <DTYPE_FLOAT64_t>np.min(image)
    max_val = <DTYPE_FLOAT64_t>np.max(image)
    for i in range(len(label_img)):
        l = label_img[i]

        # label 0 (background) is not of interest
        if l==0:
            continue

        value = <DTYPE_FLOAT64_t>image[i]

        # initialization of the criteria vector
        area_vec[l].equivalent_label = l
        area_vec[l].area = 0
        area_vec[l].stop_level = max_val
        area_vec[l].val_of_min = value

        # queue initialization
        elem.value = value
        elem.age = 0
        elem.index = i
        elem.source = i
        heappush(hp, &elem)

    level_before = min_val
    area_vec[0].equivalent_label = 0

    while hp.items > 0:
        heappop(hp, &elem)

        value = elem.value

        # check all lakes and determine the stop levels if needed.
        if value > level_before + eps:
            for l in range(number_of_minima):
                equ_l = area_vec[l].equivalent_label
                while equ_l != area_vec[equ_l].equivalent_label:
                    equ_l = area_vec[equ_l].equivalent_label
                if (area_vec[equ_l].area >= area_threshold and 
                    area_vec[l].stop_level >= max_val):
                    area_vec[l].stop_level = level_before
            level_before = elem.value

        if label_img[elem.index] and elem.index != elem.source:
            # non-marker, already visited from another neighbor
            continue

        # we find the label of the dominating lake
        label1 = label_img[elem.source]
        while label1 != area_vec[label1].equivalent_label:
            label1 = area_vec[label1].equivalent_label

        # if the criterion is met, nothing happens
        if area_vec[label1].stop_level < max_val:
            continue

        # the non-labeled pixel from the queue is marked with the
        # value of the dominating lake of its source label.
        label_img[elem.index] = label1

        # The corresponding lake is updated.
        #area_vec[label1].area += 1
        _update_area(&area_vec[label1])

        for i in range(nneighbors):
            # get the flattened address of the neighbor
            index = structure[i] + elem.index

            if not mask[index]:
                # neighbor is not in mask
                continue

            label2 = label_img[index]

            if label2:
                # neighbor has a label

                # find the label of the dominating lake
                while label2 != area_vec[label2].equivalent_label:
                    label2 = area_vec[label2].equivalent_label

                # if the label of the neighbor is different
                # from the label of the pixel taken from the queue,
                # the latter takes the WSL label.
                if label1 != label2:
                    # fusion of two lakes: the bigger eats the smaller one.

                     if area_vec[label1].area >= area_vec[label2].area:
                         area_vec[label1].area += area_vec[label2].area
                         area_vec[label2].equivalent_label = label1
                     else:
                         area_vec[label2].area += area_vec[label1].area
                         area_vec[label1].equivalent_label = label2

                # the neighbor is not added to the queue.
                continue

            # the neighbor has no label yet.
            # it is therefore added to the queue.
            age += 1
            new_elem.value = <DTYPE_FLOAT64_t>image[index]
            if compactness > 0:
                new_elem.value += <DTYPE_FLOAT64_t>(compactness *
                                   euclid_dist(index, elem.source, strides))
            new_elem.age = age
            new_elem.index = index
            new_elem.source = elem.source

            heappush(hp, &new_elem)

    heap_done(hp)

    for i in range(len(label_img)):
        label1 = label_img[i]
        if label_img[i] > 0:
            output[i] = <dtype_t>area_vec[label1].stop_level
        else:
            output[i] = image[i]

    #free(stop_level)
    #free(value_of_min)
    free(area_vec)

def area_closing(dtype_t[::1] image, #DTYPE_FLOAT64_t[::1] image,
                 DTYPE_UINT64_t area_threshold,
                 DTYPE_UINT64_t[::1] label_img,
                 DTYPE_INT32_t[::1] structure,
                 DTYPE_BOOL_t[::1] mask,
                 np.int32_t[::1] strides,
                 DTYPE_FLOAT64_t eps,
                 np.double_t compactness,
                 dtype_t[::1] output, #DTYPE_FLOAT64_t[::1] output,
                 ):
    _criteria_closing(image,
                      area_threshold,
                      label_img,
                      structure,
                      mask,
                      strides,
                      eps,
                      compactness,
                      output, 
                      _update_area)

