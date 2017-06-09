"""_criteria.pyx - flooding algorithm for criteria based openings and closings

The implementation is inspired by the implementation of the watershed
transformation (watershed.pyx) and is also based on the implementation
of the hierarchical queue provided in heap_general.pyx
"""

import numpy as np

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

cdef inline DTYPE_UINT64_t uint64_max(DTYPE_UINT64_t a, DTYPE_UINT64_t b): return a if a >= b else b
cdef inline DTYPE_UINT64_t uint64_min(DTYPE_UINT64_t a, DTYPE_UINT64_t b): return a if a <= b else b


# PropertyContainer is the base container for region properties used in the flooding
# process. 
cdef class PropertyContainer:
    cdef:
        DTYPE_UINT64_t * equivalent_label
        DTYPE_FLOAT64_t * stop_level
        DTYPE_FLOAT64_t * val_of_min

        # number of minima
        DTYPE_UINT64_t number_of_minima

        # maximal value of the image (initialization of stop_levels)
        DTYPE_FLOAT64_t max_val

    def __cinit__(self, 
                  DTYPE_UINT64_t number_of_minima,
                  DTYPE_FLOAT64_t max_val,
                  *argv,  **kwargs):
        self.number_of_minima = number_of_minima
        self.max_val = max_val

        # stores the node with which a region is fused
        self.equivalent_label = <DTYPE_UINT64_t *>malloc(sizeof(DTYPE_UINT64_t) * number_of_minima)

        # stop_level stores the level at which the flooding is to be stopped
        self.stop_level = <DTYPE_FLOAT64_t *>malloc(sizeof(DTYPE_FLOAT64_t) * number_of_minima)

        # the value of the minimum
        self.val_of_min = <DTYPE_FLOAT64_t *>malloc(sizeof(DTYPE_FLOAT64_t) * number_of_minima)

    def __dealloc__(self):
        if self.equivalent_label != NULL :
            free(self.equivalent_label)
        if self.stop_level != NULL :
            free(self.stop_level)
        if self.val_of_min != NULL :
            free(self.val_of_min)

    cpdef void flooding_postprocessing(self,
                                       DTYPE_UINT64_t label,
                                       DTYPE_FLOAT64_t value,
                                       DTYPE_FLOAT64_t level_before):
        return

    # for a given label, get the representative after fusions (after fusions
    # only one label is kept. 
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



# AreaContainer: stores the surface of the regions.
# Fusion and stop levels are with respect to the region size (number of pixels).
# Filtering: area_threshold indicates the minimal surface of extrema that remain. 
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
        
    def __dealloc__(self):
        if self.area != NULL :
            free(self.area)

    cpdef void update(self, DTYPE_UINT64_t label, DTYPE_UINT64_t index, DTYPE_FLOAT64_t value):
        self.area[label] += 1
        return

    cpdef void initialize(self, DTYPE_UINT64_t label, DTYPE_FLOAT64_t value):
        self.area[label] = 0
        self.equivalent_label[label] = label
        self.val_of_min[label] = value
        self.stop_level[label] = self.max_val
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

    cpdef void fusion(self, DTYPE_UINT64_t label1, DTYPE_UINT64_t label2):
        if self.area[label1] >= self.area[label2]:
            self.area[label1] += self.area[label2]
            self.equivalent_label[label2] = label1
        else:
            self.area[label2] += self.area[label1]
            self.equivalent_label[label1] = label2
        return



# DiameterContainer: stores the maximal extension of the regions.
# Fusion and stop levels are with respect to the maximal extension of the region.
# Filtering: diameter_threshold indicates the maximal extension of extrema that remain. 
cdef class DiameterContainer(PropertyContainer):
    cdef:
        DTYPE_UINT64_t * max_extension
        DTYPE_UINT64_t * max_coordinates
        DTYPE_UINT64_t * min_coordinates

        DTYPE_UINT64_t[:] point
        cnp.int32_t[:] image_strides
        DTYPE_UINT64_t number_of_dimensions

        # diameter threshold
        DTYPE_UINT64_t diameter_threshold

    def __cinit__(self, 
                  DTYPE_UINT64_t number_of_minima,
                  DTYPE_FLOAT64_t max_val,
                  DTYPE_UINT64_t diameter_threshold,
                  cnp.int32_t[::1] image_strides):

        # threshold for the filtering
        self.diameter_threshold = diameter_threshold

        # information for coordinate transformation (index in 1D array -> nD coordinates)
        self.number_of_dimensions = image_strides.shape[0]
        self.image_strides = image_strides

        # arrays to store the region properties
        self.max_extension = <DTYPE_UINT64_t *>malloc(sizeof(DTYPE_UINT64_t) * 
                                                      number_of_minima)
        self.max_coordinates = <DTYPE_UINT64_t *>malloc(sizeof(DTYPE_UINT64_t) * 
                                                        number_of_minima * 
                                                        self.number_of_dimensions)
        self.min_coordinates = <DTYPE_UINT64_t *>malloc(sizeof(DTYPE_UINT64_t) * 
                                                        number_of_minima * 
                                                        self.number_of_dimensions)

        # single point coordinates
        point = np.zeros(self.number_of_dimensions, dtype=np.uint64)
        self.point = point

    def __dealloc__(self):
        if self.max_extension != NULL :
            free(self.max_extension)
        if self.max_coordinates != NULL :
            free(self.max_coordinates)
        if self.min_coordinates != NULL :
            free(self.min_coordinates)

    # conversion of a 1D index to a set of coordinates. 
    cpdef void get_point(self, DTYPE_UINT64_t index):
        cdef DTYPE_UINT64_t curr_index = index

        # get coordinates from 1D index
        for i in range(self.number_of_dimensions):
            self.point[i] = curr_index // self.image_strides[i]
            curr_index = curr_index % self.image_strides[i]

        return

    cpdef void update(self, DTYPE_UINT64_t label, DTYPE_UINT64_t index, DTYPE_FLOAT64_t value):
        cdef DTYPE_UINT64_t[::1] coordinates
        cdef int i = 0
        cdef int start_index = self.number_of_dimensions * label
        cdef int stop_index = start_index + self.number_of_dimensions
        cdef int j = 0

        # get coordinates from 1D index and store them in self.point
        self.get_point(index)

        for i in range(start_index, stop_index):
            self.max_coordinates[i] = uint64_max(self.point[j], self.max_coordinates[i])
            self.min_coordinates[i] = uint64_min(self.point[j], self.min_coordinates[i])
            j += 1

        for i in range(start_index, stop_index):
            self.max_extension[label] = uint64_max(self.max_extension[label],
                                                   (self.max_coordinates[i] - self.min_coordinates[i] + 1))
        return

    cpdef void initialize(self, DTYPE_UINT64_t label, DTYPE_FLOAT64_t value):
        cdef int i = 0
        self.max_extension[label] = 0
        for i in range(self.number_of_dimensions):
            self.min_coordinates[label*self.number_of_dimensions + i] = np.max(self.image_strides)
            self.max_coordinates[label*self.number_of_dimensions + i] = 0

        self.equivalent_label[label] = label
        self.val_of_min[label] = value
        self.stop_level[label] = self.max_val
        return

    cpdef void set_stop_level(self, DTYPE_UINT64_t label,
                              DTYPE_FLOAT64_t value,
                              DTYPE_FLOAT64_t level_before):
        cdef DTYPE_UINT64_t equ_l = self.get_equivalent_label(label)
        if (self.max_extension[equ_l] > self.diameter_threshold and 
            self.stop_level[label] >= self.max_val):
            self.stop_level[label] = level_before
        return

    cpdef DTYPE_BOOL_t is_complete(self, DTYPE_UINT64_t label):
        return(self.stop_level[label] < self.max_val)

    cpdef void fusion(self, DTYPE_UINT64_t label1, DTYPE_UINT64_t label2):
        cdef int i=0
        cdef DTYPE_UINT64_t label1_c = label1 * self.number_of_dimensions
        cdef DTYPE_UINT64_t label2_c = label2 * self.number_of_dimensions
        
        if self.max_extension[label1] >= self.max_extension[label2]:
            for i in range(self.number_of_dimensions): 
                self.max_coordinates[label1_c + i] = uint64_max(self.max_coordinates[label1_c + i],
                                                              self.max_coordinates[label2_c + i])
                self.min_coordinates[label1_c + i] = uint64_min(self.min_coordinates[label1_c + i],
                                                              self.min_coordinates[label2_c + i])
                self.max_extension[label1] = uint64_max(self.max_extension[label1], 
                                                        (self.max_coordinates[label1_c + i] - 
                                                         self.min_coordinates[label1_c + i] + 1))
            self.equivalent_label[label2] = label1
        else:
            for i in range(self.number_of_dimensions): 
                self.max_coordinates[label2_c + i] = uint64_max(self.max_coordinates[label1_c + i],
                                                              self.max_coordinates[label2_c + i])
                self.min_coordinates[label2_c + i] = uint64_min(self.min_coordinates[label1_c + i],
                                                              self.min_coordinates[label2_c + i])
                self.max_extension[label2] = uint64_max(self.max_extension[label2], 
                                                        (self.max_coordinates[label2_c + i] - 
                                                         self.min_coordinates[label2_c + i] + 1))
            self.equivalent_label[label1] = label2
        return


cdef class VolumeContainer(PropertyContainer):
    cdef:
        DTYPE_UINT64_t * area
        DTYPE_FLOAT64_t * volume

        # volume threshold
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

    cpdef void flooding_postprocessing(self,
                                       DTYPE_UINT64_t label,
                                       DTYPE_FLOAT64_t value,
                                       DTYPE_FLOAT64_t level_before):
        self.volume[label] += (value - level_before) * self.area[label]
        return

    cpdef void set_stop_level(self, DTYPE_UINT64_t label,
                              DTYPE_FLOAT64_t value,
                              DTYPE_FLOAT64_t level_before):
        cdef DTYPE_UINT64_t equ_l = self.get_equivalent_label(label)
        cdef DTYPE_FLOAT64_t stop_level = self.max_val
        if self.val_of_min[label] > level_before:
            return
        if self.area[equ_l] > 0:
            stop_level = (<DTYPE_FLOAT64_t>self.volume_threshold - 
                          <DTYPE_FLOAT64_t>self.volume[equ_l]) / <DTYPE_FLOAT64_t>self.area[equ_l] + <DTYPE_FLOAT64_t>level_before
            if stop_level < level_before:
                stop_level = level_before
        if (stop_level <= value and self.stop_level[label] >= self.max_val):
            self.stop_level[label] = stop_level
        return

    cpdef DTYPE_BOOL_t is_complete(self, DTYPE_UINT64_t label):
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

@cython.boundscheck(False)
def _criteria_closing(dtype_t[::1] image,
                      DTYPE_UINT64_t[::1] label_img,
                      DTYPE_INT32_t[::1] structure,
                      DTYPE_BOOL_t[::1] mask,
                      cnp.int32_t[::1] strides,
                      DTYPE_FLOAT64_t eps,
                      dtype_t[::1] output,
                      property_class,
                      ):
    """Perform criteria based closings.

    Parameters
    ----------

    image : array of arbitrary type
        The flattened image pixels.
    label_img : array of int
        labeled local minima of image.
    structure : array of int
        A list of coordinate offsets to compute the raveled coordinates of each
        neighbor from the raveled coordinates of the current pixel.
    mask : array of int
        An array of the same shape as `image` where each pixel contains a
        nonzero value if it is to be considered for the filtering.
        NOTE: it is *essential* that the border pixels (those
        with neighbors falling outside the volume) are all set to zero, or
        segfaults could occur.
    strides : array of int
        An array representing the number of steps to move along each dimension.
        This is used to transform raveled indices into coordinates.
    eps : floating point
        This parameter determines the smallest flooding step. Modifying this parameter
        makes only sense for floating point images and if the function turns out to be
        slow. Larger step sizes lead to speed increase.
    output : array of int
        The output array, which must already contain nonzero entries at all the
        seed locations.
    property_class : cython class derived from PropertyContainer.
        This class determines the criterion according to which regions are fused and
        filtering is performed. 
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

    cdef bint is_float = False
    if dtype_t is np.float32_t or dtype_t is np.float64_t:
        is_float = True

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

    while hp.items > 0:
        heappop(hp, &elem)

        if label_img[elem.index] and elem.index != elem.source:
            # non-marker, already visited from another neighbor
            continue

        value = elem.value

        # check all lakes and determine the stop levels if needed.
        if value > level_before + eps:
            # evalue the criterion and set the stop levels
            for l in range(1, number_of_minima):
                property_class.set_stop_level(l, value, level_before)

            # some post-processing step after having set the stop levels.
            for l in range(1, number_of_minima):
                property_class.flooding_postprocessing(l, value, level_before)

            level_before = value

        # we find the label of the dominating lake
        label1 = property_class.get_equivalent_label(label_img[elem.source])

        # the non-labeled pixel from the queue is marked with the
        # value of the dominating lake of its source label.
        label_img[elem.index] = label_img[elem.source]
        #label_img[elem.index] = label1

        # if the criterion is not met, the final label image is updated.
        # regions that fulfill already the criterion do not grow anymore.
        if not property_class.is_complete(label_img[elem.source]):
            label_final[elem.index] = label_img[elem.index]
            #label_final[elem.index] = label1

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

                    # In this case, the equivalent labels might have changed
                    # according to the fusion rule defined in the property_class
                    # label1 therefore needs to be set again (label2 is now obsolete).
                    label1 = property_class.get_equivalent_label(label1)

                # the neighbor is not added to the queue.
                continue

            # the neighbor has no label yet.
            # it is therefore added to the queue.
            age += 1
            new_elem.value = <DTYPE_FLOAT64_t>image[index]
            new_elem.age = age
            new_elem.index = index
            new_elem.source = elem.source

            heappush(hp, &new_elem)

    heap_done(hp)

    # After the flooding, the stop levels are set.
    for i in range(len(label_img)):
        label1 = label_final[i]
        if label1 > 0:
            output[i] = <dtype_t>property_class.get_stop_level(<DTYPE_UINT64_t>label1)
            if not is_float:
                # This corresponds to the "at least" in volume filling, i.e. if there is 
                # an in between value, the larger fill level is chosen. 
                if <DTYPE_FLOAT64_t>output[i] - property_class.get_stop_level(<DTYPE_UINT64_t>label1) < 0:
                    output[i] += 1
        else:
            output[i] = image[i]

    return

# area closing fills all minima until all local minima have at least an area (number of pixels)
# of area_threshold. 
def area_closing(dtype_t[::1] image,
                 DTYPE_UINT64_t area_threshold,
                 DTYPE_UINT64_t[::1] label_img,
                 DTYPE_INT32_t[::1] structure,
                 DTYPE_BOOL_t[::1] mask,
                 cnp.int32_t[::1] strides,
                 DTYPE_FLOAT64_t eps,
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
                      output, 
                      area_container)

# diameter closing fills all minima until the corresponding catchment bassins 
# have at least a maximal extension of diameter_threshold. 
def diameter_closing(dtype_t[::1] image,
                     DTYPE_UINT64_t diameter_threshold,
                     DTYPE_UINT64_t[::1] label_img,
                     DTYPE_INT32_t[::1] structure,
                     DTYPE_BOOL_t[::1] mask,
                     cnp.int32_t[::1] strides,
                     DTYPE_FLOAT64_t eps,
                     dtype_t[::1] output,
                     ):

    cdef int number_of_minima = np.max(label_img) + 1
    cdef DTYPE_FLOAT64_t max_val = <DTYPE_FLOAT64_t>np.max(image)

    # The specialization of the PropertyContainer
    cdef DiameterContainer diameter_container = DiameterContainer(number_of_minima,
                                                                  max_val,
                                                                  diameter_threshold, 
                                                                  strides)

    _criteria_closing(image,
                      label_img,
                      structure,
                      mask,
                      strides,
                      eps,
                      output, 
                      diameter_container)

# volume_fill fills the image such that for each minimum the volume that has
# been added on top of the image has at least a volume (integral of intensities)
# of volume_threshold.
def volume_fill(dtype_t[::1] image,
                DTYPE_FLOAT64_t volume_threshold,
                DTYPE_UINT64_t[::1] label_img,
                DTYPE_INT32_t[::1] structure,
                DTYPE_BOOL_t[::1] mask,
                cnp.int32_t[::1] strides,
                DTYPE_FLOAT64_t eps,
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
                      output, 
                      volume_container)
