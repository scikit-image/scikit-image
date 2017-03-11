"""_criteria.pyx - flooding algorithm for criteria based openings and closings

The implementation is inspired by the implementation of the watershed
transformation (watershed.pyx) and is also based on the implementation
of the hierarchical queue provided in heap_general.pyx
"""
#import numpy as np
#from IPython.html.widgets.interaction import _get_min_max_value
#from ._watershed import _euclid_dist

cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_FLOAT64_t
ctypedef cnp.int32_t DTYPE_INT32_t
ctypedef cnp.uint32_t DTYPE_UINT32_t
ctypedef cnp.uint64_t DTYPE_UINT64_t
ctypedef cnp.int8_t DTYPE_BOOL_t


include "heap_watershed.pxi"
include "util.pxi"

cdef struct Area:
    DTYPE_UINT64_t area
    DTYPE_UINT64_t equivalent_label
    DTYPE_FLOAT64_t value_of_min
    DTYPE_FLOAT64_t stop_level


@cython.boundscheck(False)
def area_closing(DTYPE_FLOAT64_t[::1] image,
                 DTYPE_UINT32_t area_threshold,
                 DTYPE_UINT64_t[::1] label_img,
                 DTYPE_INT32_t[::1] structure,
                 DTYPE_BOOL_t[::1] mask,
                 cnp.int32_t[::1] strides,
                 DTYPE_FLOAT64_t eps,
                 cnp.double_t compactness,
                 DTYPE_FLOAT64_t[::1] output,
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
    cdef DTYPE_FLOAT64_t level_before=0.0;

    # hierarchical queue for the flooding
    cdef Heap *hp = <Heap *> heap_from_numpy2()

    # array for characterization of minima
    cdef int number_of_minima = label_img.max() + 1
    cdef Area* area_array = <Area *>malloc(sizeof(Area) * number_of_minima)
    cdef Area[:] area_vec = <Area[:number_of_minima]> area_array

    min_val = cnp.min(image)
    max_val = cnp.max(image)
    for i in range(len(label_img)):
        l = label_img[i]

        # label 0 (background) is not of interest
        if label1==0:
            continue

        value = image[i]

        # initialization of the criteria vector
        area_vec[l].area += 1
        area_vec[l].equivalence = l
        area_vec[l].value_of_min = value

        # queue initialization
        elem.value = value
        elem.age = 0
        elem.index = i
        elem.source = i
        heappush(hp, &elem)

    for l in range(1, number_of_minima):
        if area_vec[l].area >= area_threshold:
            area_vec[l].stop_level = area_vec[l].value_of_min
        else:
            area_vec[l].stop_level = max_val

    level_before = min_val
    area_vec[0].equivalence = 0

    while hp.items > 0:
        heappop(hp, &elem)

        # check all lakes and determine the stop levels if needed.
        if elem.value > level_before + eps:
            for l in range(number_of_minima):
                equ_l = area_vec[l].equivalence
                while equ_l != area_vec[equ_l].equivalence:
                    equ_l = area_vec[equ_l].equivalence
                if (area_vec[equ_l].area >= area_threshold and area_vec[l].stop_level >= max_val):
                    area_vec[l].stop_level = level_before
            level_before = elem.value

        if label_img[elem.index] and elem.index != elem.source:
            # non-marker, already visited from another neighbor
            continue

        # we find the label of the dominating lake
        label1 = label_img[elem.source]
        while label1 != area_vec[label1].equivalence:
            label1 = area_vec[label1].equivalence

        # the non-labeled pixel from the queue is marked with the
        # value of the domainting lake of its source label.
        label_img[elem.index] = label1

        # The corresponding lake is updated.
        area_vec[label1].area += 1

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
                while label2 != area_vec[label2].equivalence:
                    label2 = area_vec[label2].equivalence

                # if the label of the neighbor is different
                # from the label of the pixel taken from the queue,
                # the latter takes the WSL label.
                if label1 != label2:
                    # fusion of two lakes: the bigger eats the smaller one.
                    if area_vec[label1].area >= area_vec[label2].area:
                        area_vec[label1].area += area_vec[label2].area
                        area_vec[label2].equivalence = label1
                    else:
                        area_vec[label2].area += area_vec[label1].area
                        area_vec[label1].equivalence = label2

                # the neighbor is not added to the queue.
                continue

            # the neighbor has no label yet.
            # it is therefore added to the queue.
            age += 1
            new_elem.value = image[index]
            if compactness > 0:
                new_elem.value += (compactness *
                                   euclid_dist(index, elem.source, strides))
            new_elem.age = age
            new_elem.index = index
            new_elem.source = elem.source

            heappush(hp, &new_elem)

    heap_done(hp)
    
    for i in range(len(label_img)):
        label1 = label_img[i]
        if label_img[i]:
            output[i] = area_vec[label1].stop_level
        else:
            output[i] = image[i]


