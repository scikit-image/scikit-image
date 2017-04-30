"""_criteria.pyx - flooding algorithm for criteria based openings and closings

The implementation is inspired by the implementation of the watershed
transformation (watershed.pyx) and is also based on the implementation
of the hierarchical queue provided in heap_general.pyx
"""

import numpy as np
from libc.math cimport sqrt

import skimage.io
import os
from libc.stdlib cimport free, malloc, realloc

cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_FLOAT64_t
ctypedef np.int32_t DTYPE_INT32_t
ctypedef np.uint32_t DTYPE_UINT32_t
ctypedef np.uint64_t DTYPE_UINT64_t
ctypedef np.int64_t DTYPE_INT64_t
ctypedef np.uint8_t DTYPE_BOOL_t
ctypedef np.uint8_t DTYPE_UINT8_t


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

def print_image(dtype_t[::1] image, int width):
    print
    tempStr = ''
    for i in range(len(image)):
        tempStr += " %3i" % image[i]
        if (i+1)%width == 0:
            print tempStr
            tempStr = ''
    print

cdef class PropertyContainer:
    cdef:
        DTYPE_UINT64_t * equivalent_label
        DTYPE_FLOAT64_t * stop_level
        DTYPE_FLOAT64_t * val_of_min

        # number of minima
        DTYPE_UINT64_t number_of_minima

        DTYPE_FLOAT64_t max_val

    def __cinit__(self, 
                  DTYPE_UINT64_t number_of_minima,
                  DTYPE_FLOAT64_t max_val,
                  *argv,  **kwargs):
        self.number_of_minima = number_of_minima
        self.max_val = max_val

        self.equivalent_label = <DTYPE_UINT64_t *>malloc(sizeof(DTYPE_UINT64_t) * number_of_minima)
        self.stop_level = <DTYPE_FLOAT64_t *>malloc(sizeof(DTYPE_FLOAT64_t) * number_of_minima)
        self.val_of_min = <DTYPE_FLOAT64_t *>malloc(sizeof(DTYPE_FLOAT64_t) * number_of_minima)
        print 'PropertyContainer init: %.2f' % max_val 


    def __dealloc__(self):
        if self.equivalent_label != NULL :
            free(self.equivalent_label)
        if self.stop_level != NULL :
            free(self.stop_level)
        if self.val_of_min != NULL :
            free(self.val_of_min)

    cpdef DTYPE_UINT64_t get_equivalent_label(self, DTYPE_UINT64_t label):
        while label != self.equivalent_label[label]:
            label = self.equivalent_label[label]
        return label

    cpdef DTYPE_UINT64_t get_number_of_minima(self):
        return(self.number_of_minima)

    cpdef DTYPE_FLOAT64_t get_stop_level(self, DTYPE_UINT64_t label):
        return self.stop_level[label]

    cpdef DTYPE_FLOAT64_t get_max_val(self):
        return(self.max_val)


cdef class AreaContainer(PropertyContainer):
    cdef:
        DTYPE_UINT64_t * area

        # area threshold
        DTYPE_UINT64_t area_threshold

    def __cinit__(self, 
                  DTYPE_UINT64_t number_of_minima,
                  DTYPE_FLOAT64_t max_val,
                  DTYPE_UINT64_t area_threshold):
        self.area_threshold = area_threshold
        self.area = <DTYPE_UINT64_t *>malloc(sizeof(DTYPE_UINT64_t) * number_of_minima)
        print 'AreaContainer init: %.2f' % max_val 
        
    def __dealloc__(self):
        if self.area != NULL :
            free(self.area)

    cpdef void update(self, DTYPE_UINT64_t label, DTYPE_UINT64_t index, DTYPE_FLOAT64_t value):
        self.area[label] += 1
        return

    cpdef void output_all(self):
        cdef int i=0
        for i in range(1, self.number_of_minima): 
            print 'label %i\tarea: %i\tstop_level: %.2f' % (i, self.area[i], self.stop_level[i])
        return
    
    cpdef void initialize(self, DTYPE_UINT64_t label, DTYPE_FLOAT64_t value):
        self.area[label] = 0
        self.equivalent_label[label] = label
        self.val_of_min[label] = value
        self.stop_level[label] = self.max_val
        print 'during initialization: label = %i\tstop_level = %.2f\tmax = %.2f' % (label, self.stop_level[label], self.max_val)
        return

    cpdef void set_stop_level(self, DTYPE_UINT64_t label,
                              DTYPE_FLOAT64_t value,
                              DTYPE_FLOAT64_t level_before):
        cdef DTYPE_UINT64_t equ_l = self.get_equivalent_label(label)
        if (self.area[equ_l] >= self.area_threshold and 
            self.stop_level[label] >= self.max_val):
            self.stop_level[label] = level_before
        return
    
    cpdef DTYPE_BOOL_t is_complete(self, DTYPE_UINT64_t label):
        return(self.stop_level[label] < self.max_val)
        #return (self.area[label] >= self.area_threshold)
    
    cpdef void fusion(self, DTYPE_UINT64_t label1, DTYPE_UINT64_t label2):
        if self.area[label1] >= self.area[label2]:
            self.area[label1] += self.area[label2]
            self.equivalent_label[label2] = label1
        else:
            self.area[label2] += self.area[label1]
            self.equivalent_label[label1] = label2
        return

cdef class VolumeContainer(PropertyContainer):
    cdef:
        DTYPE_UINT64_t * area
        DTYPE_FLOAT64_t * volume

        # area threshold
        DTYPE_FLOAT64_t volume_threshold

    def __cinit__(self, 
                  DTYPE_UINT64_t number_of_minima, 
                  DTYPE_FLOAT64_t max_val,
                  DTYPE_FLOAT64_t volume_threshold):
        self.volume_threshold = volume_threshold
        self.volume = <DTYPE_FLOAT64_t *>malloc(sizeof(DTYPE_FLOAT64_t) * number_of_minima)
        self.area = <DTYPE_UINT64_t *>malloc(sizeof(DTYPE_UINT64_t) * number_of_minima)
        
    def __dealloc__(self):
        if self.area != NULL :
            free(self.area)
        if self.volume != NULL :
            free(self.volume)

    cpdef void update(self, DTYPE_UINT64_t label, DTYPE_UINT64_t index, DTYPE_FLOAT64_t value):
        self.area[label] += 1
        return

    cpdef void initialize(self, DTYPE_UINT64_t label, DTYPE_FLOAT64_t value):
        self.volume[label] = 0
        self.area[label] = 0
        self.equivalent_label[label] = label
        self.val_of_min[label] = value
        self.stop_level[label] = self.max_val
        return

    cpdef void set_stop_level(self, DTYPE_UINT64_t label,
                              DTYPE_FLOAT64_t value,
                              DTYPE_FLOAT64_t level_before):
        label = self.get_equivalent_label(label)
        self.volume[label] += (value - level_before) * self.area[label]
        if (self.volume[label] >= self.volume_threshold and 
            self.stop_level[label] >= self.max_val):
            self.stop_level[label] = level_before
        return
    
    cpdef DTYPE_BOOL_t is_complete(self, DTYPE_UINT64_t label):
        #return (self.volume[label] >= self.volume_threshold)
        return(self.stop_level[label] < self.max_val)
    
    cpdef void fusion(self, DTYPE_UINT64_t label1, DTYPE_UINT64_t label2):
        if self.volume[label1] >= self.volume[label2]:
            self.volume[label1] += self.volume[label2]
            self.area[label1] += self.area[label2]
            self.equivalent_label[label2] = label1
        else:
            self.volume[label2] += self.volume[label1]
            self.area[label2] += self.area[label1]
            self.equivalent_label[label1] = label2
        return

cdef class BackupAreaContainer:
    cdef:
        DTYPE_UINT64_t * area
        DTYPE_UINT64_t * equivalent_label
        DTYPE_FLOAT64_t * stop_level
        DTYPE_FLOAT64_t * val_of_min

        # number of minima
        DTYPE_UINT64_t number_of_minima

        # area threshold
        DTYPE_UINT64_t area_threshold

        DTYPE_FLOAT64_t max_val

    def __cinit__(self, 
                  DTYPE_UINT64_t number_of_minima, 
                  DTYPE_UINT64_t area_threshold, 
                  DTYPE_FLOAT64_t max_val):
        self.number_of_minima = number_of_minima
        self.area_threshold = area_threshold
        self.area = <DTYPE_UINT64_t *>malloc(sizeof(DTYPE_UINT64_t) * number_of_minima)
        self.equivalent_label = <DTYPE_UINT64_t *>malloc(sizeof(DTYPE_UINT64_t) * number_of_minima)
        self.stop_level = <DTYPE_FLOAT64_t *>malloc(sizeof(DTYPE_FLOAT64_t) * number_of_minima)
        self.val_of_min = <DTYPE_FLOAT64_t *>malloc(sizeof(DTYPE_FLOAT64_t) * number_of_minima)
        self.max_val = max_val

    def __dealloc__(self):
        if self.area != NULL :
            free(self.area)
        if self.equivalent_label != NULL :
            free(self.equivalent_label)
        if self.stop_level != NULL :
            free(self.stop_level)
        if self.val_of_min != NULL :
            free(self.val_of_min)

    cpdef void update(self, DTYPE_UINT64_t label):
        self.area[label] += 1
        return

    cpdef void initialize(self, DTYPE_UINT64_t label, DTYPE_FLOAT64_t value):
        self.area[label] = 0
        self.equivalent_label[label] = label
        self.val_of_min[label] = value
        self.stop_level[label] = self.max_val 
        return

    cpdef DTYPE_UINT64_t get_equivalent_label(self, DTYPE_UINT64_t label):
        while label != self.equivalent_label[label]:
            label = self.equivalent_label[label]
        return label

    cpdef DTYPE_UINT64_t get_number_of_minima(self):
        return(self.number_of_minima)

    cpdef DTYPE_FLOAT64_t get_stop_level(self, DTYPE_UINT64_t label):
        return self.stop_level[label]

    cpdef void set_stop_level(self, DTYPE_UINT64_t label, DTYPE_FLOAT64_t level_before):
        label = self.get_equivalent_label(label)
        if (self.area[label] >= self.area_threshold and 
            self.stop_level[label] >= self.max_val):
            self.stop_level[label] = level_before
        return
    
    cpdef DTYPE_BOOL_t is_complete(self, DTYPE_UINT64_t label):
        return (self.area[label] >= self.area_threshold)
    
    cpdef void fusion(self, DTYPE_UINT64_t label1, DTYPE_UINT64_t label2):
        if self.area[label1] >= self.area[label2]:
            self.area[label1] += self.area[label2]
            self.equivalent_label[label2] = label1
        else:
            self.area[label2] += self.area[label1]
            self.equivalent_label[label1] = label2
        return


@cython.boundscheck(False)
def _criteria_closing(dtype_t[::1] image, #DTYPE_FLOAT64_t[::1] image,
                      DTYPE_UINT64_t[::1] label_img,
                      DTYPE_INT32_t[::1] structure,
                      DTYPE_BOOL_t[::1] mask,
                      np.int32_t[::1] strides,
                      DTYPE_FLOAT64_t eps,
                      np.double_t compactness,
                      dtype_t[::1] output, #DTYPE_FLOAT64_t[::1] output,
                      property_class,
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
    cdef DTYPE_UINT64_t label1 = 0
    cdef DTYPE_UINT64_t label2 = 0

    cdef DTYPE_FLOAT64_t min_val;
    cdef DTYPE_FLOAT64_t value;
    cdef DTYPE_FLOAT64_t level_before;

    # hierarchical queue for the flooding
    cdef Heap *hp = <Heap *> heap_from_numpy2()

    # number of minima is already stored in the property_class.
    cdef int number_of_minima = property_class.get_number_of_minima()

    cdef DTYPE_UINT64_t[::1] label_final = label_img.copy()

    min_val = <DTYPE_FLOAT64_t>np.min(image)
    for i in range(len(label_img)):
        l = label_img[i]

        # label 0 (background) is not of interest
        if l==0:
            continue

        value = <DTYPE_FLOAT64_t>image[i]

        # initialization of the criteria vector
        property_class.initialize(l, value)

        # queue initialization
        elem.value = value
        elem.age = 0
        elem.index = i
        elem.source = i
        heappush(hp, &elem)

    level_before = min_val
    property_class.initialize(0, 0.0)

    print
    print 'INITIALIZATION'
    property_class.output_all()
    
    
    while hp.items > 0:
        heappop(hp, &elem)

        if label_img[elem.index] and elem.index != elem.source:
            # non-marker, already visited from another neighbor
            continue

        value = elem.value

        # check all lakes and determine the stop levels if needed.
        if value > level_before + eps:
            print
            print 'STEP: ', level_before, ' --> ', value
            
            for l in range(number_of_minima):
                property_class.set_stop_level(l, value, level_before)
            property_class.output_all()
            level_before = value

            print
            print 'LABEL_IMG'
            print
            print_image(label_img, 14)
            print
            print 'LABEL_FINAL'
            print
            print_image(label_final, 14)

            
        # we find the label of the dominating lake
        label1 = property_class.get_equivalent_label(label_img[elem.source])

        # if the criterion is met, nothing happens
#        if property_class.is_complete(label1):
#            continue

        # the non-labeled pixel from the queue is marked with the
        # value of the dominating lake of its source label.
        #label_img[elem.index] = label1
        label_img[elem.index] = label_img[elem.source]

        # if the criterion is not met, the final label image is updated.
        # regions that fulfill already the criterion do not grow anymore.
        if not property_class.is_complete(label1):
            # label_final[elem.index] = label1
            label_final[elem.index] = label_img[elem.source]

        # The corresponding lake is updated.
        property_class.update(label1, elem.index, value)

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
                label2 = property_class.get_equivalent_label(label2)

                # if the label of the neighbor is different
                # from the label of the pixel taken from the queue,
                # the latter takes the WSL label.
                if label1 != label2:
                    # fusion of two lakes: the bigger eats the smaller one.
                    property_class.fusion(label1, label2)

                # the neighbor is not added to the queue.
                continue

            # the neighbor has no label yet.
            # it is therefore added to the queue.
            age += 1
            new_elem.value = <DTYPE_FLOAT64_t>image[index]
            if compactness > 0:
                new_elem.value += <DTYPE_FLOAT64_t>(compactness *
                                                    euclid_dist(index,
                                                                elem.source,
                                                                strides))
            new_elem.age = age
            new_elem.index = index
            new_elem.source = elem.source

            heappush(hp, &new_elem)

    heap_done(hp)

    print
    print 'BEFORE FINAL ASSIGNMENT: LABEL_FINAL'
    print
    print_image(label_final, 14)

    for i in range(len(label_img)):
        label1 = label_final[i]
        if label1 > 0:
            output[i] = <dtype_t>property_class.get_stop_level(<DTYPE_UINT64_t>label1)
        else:
            output[i] = image[i]


def area_closing(dtype_t[::1] image,
                 DTYPE_UINT64_t area_threshold,
                 DTYPE_UINT64_t[::1] label_img,
                 DTYPE_INT32_t[::1] structure,
                 DTYPE_BOOL_t[::1] mask,
                 np.int32_t[::1] strides,
                 DTYPE_FLOAT64_t eps,
                 np.double_t compactness,
                 dtype_t[::1] output,
                 ):
    
    cdef int number_of_minima = np.max(label_img) + 1
    cdef DTYPE_FLOAT64_t max_val = <DTYPE_FLOAT64_t>np.max(image)
    cdef AreaContainer area_container = AreaContainer(number_of_minima,
                                                      max_val,
                                                      area_threshold)
    
    
    _criteria_closing(image,
                      label_img,
                      structure,
                      mask,
                      strides,
                      eps,
                      compactness,
                      output, 
                      area_container)


def volume_fill(dtype_t[::1] image,
                DTYPE_FLOAT64_t volume_threshold,
                DTYPE_UINT64_t[::1] label_img,
                DTYPE_INT32_t[::1] structure,
                DTYPE_BOOL_t[::1] mask,
                np.int32_t[::1] strides,
                DTYPE_FLOAT64_t eps,
                np.double_t compactness,
                dtype_t[::1] output,
                ):
    
    cdef int number_of_minima = np.max(label_img) + 1
    cdef DTYPE_FLOAT64_t max_val = <DTYPE_FLOAT64_t>np.max(image)
    cdef VolumeContainer volume_container = VolumeContainer(number_of_minima,
                                                            max_val,
                                                            volume_threshold)

    _criteria_closing(image,
                      label_img,
                      structure,
                      mask,
                      strides,
                      eps,
                      compactness,
                      output, 
                      volume_container)
