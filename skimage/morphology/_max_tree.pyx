#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

"""_max_tree.pyx - building a max-tree from an image.

This is an implementation of the max-tree, which is a morphological
representation of the image. Many morphological operators can be built
from this representation, namely attribute openings and closings.

This file also contains implementations of max-tree based filters and
functions to characterize the tree components.
"""

import numpy as np

from libc.stdlib cimport free, malloc, realloc
from scipy.constants import point
from cProfile import label

cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_FLOAT64_t
ctypedef np.int32_t DTYPE_INT32_t
ctypedef np.uint32_t DTYPE_UINT32_t
ctypedef np.uint64_t DTYPE_UINT64_t
ctypedef np.int64_t DTYPE_INT64_t
ctypedef np.uint8_t DTYPE_BOOL_t
ctypedef np.uint8_t DTYPE_UINT8_t
ctypedef np.int8_t DTYPE_INT8_t

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


cdef DTYPE_INT64_t find_root_recursive(DTYPE_INT64_t[::1] parent,
                                       DTYPE_INT64_t index):
    """recursive function to get the root of the current tree. Importantly, the
    function changes the tree (path compression), that reduces the complexity
    from O(n*n) to O(n*log(n)). Despite of path compression, our tests showed
    that the non-recursive version seems to perform better. We leave this
    version for future improvements.
    """
    if parent[index] != index:
        parent[index] = find_root_recursive(parent, parent[index])
    return parent[index]


cdef inline DTYPE_INT64_t find_root(DTYPE_INT64_t[::1] parent,
                                    DTYPE_INT64_t index):
    """function to get the root of the current tree. Here, we do without path
    compression and accept the higher complexity, but the function is inline
    and avoids some overhead induced by its recursive version.
    """
    while parent[index] != parent[parent[index]]:
        parent[index] = parent[parent[index]]
    return parent[index]


cdef void canonize(dtype_t[::1] image, DTYPE_INT64_t[::1] parent,
                   DTYPE_INT64_t[::1] sorted_indices):
    """generates a max-tree for which every node's parent is a canonical node.
    The parent of a non-canonical pixel is a canonical pixel.
    The parent of a canonical pixel is also a canonical pixel with a different
    value. There is exactly one canonical pixel for each component in the
    component tree.
    """
    cdef DTYPE_INT64_t q = 0
    cdef DTYPE_INT64_t p
    for p in sorted_indices:
        q = parent[p]
        if image[q] == image[parent[q]]:
            parent[p] = parent[q]


cdef np.ndarray[DTYPE_INT32_t, ndim=2] unravel_offsets(
                                    DTYPE_INT32_t[::1] offsets,
                                    DTYPE_INT32_t[::1] shape):
    """Unravels a list of offset indices. These offsets can be (and normally
    are) negative. The function generates an array of shape
    (number of offsets, image dimensions), where each row corresponds
    to the coordinates of each point.

    See also
    --------
    unravel_index
    """

    cdef DTYPE_INT32_t number_of_dimensions = len(shape)
    cdef DTYPE_INT32_t number_of_points = len(offsets)
    cdef np.ndarray[DTYPE_INT32_t, ndim=2] points = np.zeros(
                                        (number_of_points,
                                         number_of_dimensions),
                                        dtype=np.int32)
    cdef DTYPE_INT32_t neg_shift = - np.min(offsets)

    cdef DTYPE_INT32_t i, offset, curr_index, coord

    center_point = np.unravel_index(neg_shift, shape)

    for i, offset in enumerate(offsets):
        current_point = np.unravel_index(offset + neg_shift, shape)
        for d in range(number_of_dimensions):
            points[i, d] = current_point[d] - center_point[d]

    return points


cdef DTYPE_UINT8_t _is_valid_neighbor(DTYPE_INT64_t index,
                                      DTYPE_INT32_t[::1] coordinates,
                                      DTYPE_INT32_t[::1] shape):
    """checks whether a neighbor of a given pixel is inside the image plane.
    The pixel is given in form of an index in a raveled array, the neighbor
    is given as a list of coordinates (offset). If the neighbor falls outside
    the image, the function gives back 0, otherwise 1.
    """

    cdef DTYPE_INT64_t number_of_dimensions = len(shape)
    cdef DTYPE_INT64_t res_coord = 0
    cdef int i = 0

    cdef np.ndarray[DTYPE_INT32_t, ndim=1] p_coord = np.array(
                                                np.unravel_index(index, shape),
                                                dtype=np.int32)

    # get the coordinates of the point from a 1D index
    for i in range(number_of_dimensions):
        res_coord = p_coord[i] + coordinates[i]

        if res_coord < 0:
            return 0
        if res_coord >= shape[i]:
            return 0

    return 1


cpdef np.ndarray[DTYPE_FLOAT64_t, ndim=1] _compute_area(dtype_t[::1] image,
                                            DTYPE_INT64_t[::1] parent,
                                            DTYPE_INT64_t[::1] sorted_indices):
    """computes the area of all max-tree components
    attribute to be used in area opening and closing
    """
    cdef DTYPE_INT64_t p_root = sorted_indices[0]
    cdef DTYPE_INT64_t p, q
    cdef DTYPE_UINT64_t number_of_pixels = len(image)
    cdef np.ndarray[DTYPE_FLOAT64_t, ndim=1] area = np.ones(number_of_pixels,
                                                              dtype=np.float64)

    for p in sorted_indices[::-1]:
        if p == p_root:
            continue
        q = parent[p]
        area[q] = area[q] + area[p]

    return area


cpdef np.ndarray[DTYPE_FLOAT64_t, ndim=1] _compute_extension(
                                            dtype_t[::1] image,
                                            DTYPE_INT32_t[::1] shape,
                                            DTYPE_INT64_t[::1] parent,
                                            DTYPE_INT64_t[::1] sorted_indices):
    """computes the bounding box extension of all max-tree components
    attribute to be used in diameter opening and closing
    """
    cdef DTYPE_INT64_t p_root = sorted_indices[0]
    cdef DTYPE_INT64_t p, q
    cdef DTYPE_UINT64_t number_of_pixels = len(image)
    cdef np.ndarray[DTYPE_FLOAT64_t, ndim=1] extension = np.ones(
                                                    number_of_pixels,
                                                    dtype=np.float64)
    cdef np.ndarray[DTYPE_FLOAT64_t, ndim=2] max_coord = np.array(
                        np.unravel_index(np.arange(number_of_pixels),
                        shape), dtype=np.float64).T
    cdef np.ndarray[DTYPE_FLOAT64_t, ndim=2] min_coord = np.array(
                        np.unravel_index(np.arange(number_of_pixels),
                        shape), dtype=np.float64).T

    for p in sorted_indices[::-1]:
        if p == p_root:
            continue
        q = parent[p]
        max_coord[q] = np.maximum(max_coord[q], max_coord[p])
        min_coord[q] = np.minimum(min_coord[q], min_coord[p])
        extension[q] = np.max(max_coord[q] - min_coord[q]) + 1

    return extension


# _max_tree_local_maxima cacluates the local maxima from the max-tree
# representation this is interesting if the max-tree representation has
# already been calculated for other reasons. Otherwise, it is not the most
# efficient method. If the parameter label is True, the minima are labeled.
cpdef void _max_tree_local_maxima(dtype_t[::1] image,
                                  DTYPE_UINT64_t[::1] output,
                                  DTYPE_INT64_t[::1] parent,
                                  DTYPE_INT64_t[::1] sorted_indices
                                  ):
    """Finds the local maxima in image from the max-tree representation.

    Parameters
    ----------

    image : array of arbitrary type
        The flattened image pixels.
    output : array of the same shape and type as image.
        The output image must contain only ones.
    parent : array of int
        Image of the same shape as the input image. The value
        at each pixel is the parent index of this pixel in the max-tree
        reprentation.
    sorted_indices : array of int
        List of length = number of pixels. Each element
        corresponds to one pixel index in the image. It encodes the order
        of elements in the tree: a parent of a pixel always comes before
        the element itself. More formally: i < j implies that j cannot be
        the parent of i.
    """

    cdef DTYPE_INT64_t p_root = sorted_indices[0]
    cdef DTYPE_INT64_t p, q
    cdef DTYPE_UINT64_t number_of_pixels = len(image)
    cdef DTYPE_UINT64_t label = 1

    for p in sorted_indices[::-1]:
        if p == p_root:
            continue

        q = parent[p]

        # if p is canonical (parent has a different value)
        if image[p] != image[q]:
            output[q] = 0

            # if output[p] was the parent of some other canonical
            # pixel, it has been set to zero. Only the leaves
            # (local maxima) are thus > 0.
            if output[p] > 0:
                output[p] = label
                label += 1

    for p in sorted_indices[::-1]:
        if p == p_root:
            continue

        q = parent[p]

        # if p is not canonical (parent has the same value)
        if image[p] == image[q]:
            # in this case we propagate the value
            output[p] = output[q]
            continue

    return


# direct filter (criteria based filter)
cpdef void _direct_filter(dtype_t[::1] image,
                          dtype_t[::1] output,
                          DTYPE_INT64_t[::1] parent,
                          DTYPE_INT64_t[::1] sorted_indices,
                          DTYPE_FLOAT64_t[::1] attribute,
                          DTYPE_FLOAT64_t attribute_threshold
                          ):
    """Direct filtering. Produces an image in which for all possible
    thresholds, each connected component has an
    attribute >= attribute_threshold. This is the basic function
    which is called by area_opening, diameter_opening, etc.
    For area_opening for instance, the attribute has to be the area.
    In this case, an image is produced for which all connected
    components for all thresholds have at least an area (pixel count)
    of the threshold given by the user.

    Parameters
    ----------

    image : array of arbitrary type
            The flattened image pixels.
    output : array of the same shape and type as image.
    parent : array of int
        Image of the same shape as the input image. The value
        at each pixel is the parent index of this pixel in the max-tree
        reprentation.
    sorted_indices : array of int
        List of length = number of pixels. Each element
        corresponds to one pixel index in the image. It encodes the order
        of elements in the tree: a parent of a pixel always comes before
        the element itself. More formally: i < j implies that j cannot be
        the parent of i.
    attribute : array of float
        Contains the attributes for the max-tree
    attribute_threshold : float
        The threshold to be applied to the attribute.
    """

    cdef DTYPE_INT64_t p_root = sorted_indices[0]
    cdef DTYPE_INT64_t p, q
    cdef DTYPE_UINT64_t number_of_pixels = len(image)

    if attribute[p_root] < attribute_threshold:
        output[p_root] = 0
    else:
        output[p_root] = image[p_root]

    for p in sorted_indices:
        if p == p_root:
            continue

        q = parent[p]

        # this means p is not canonical
        # in other words, it has a parent that has the
        # same image value.
        if image[p] == image[q]:
            output[p] = output[q]
            continue

        if attribute[p] < attribute_threshold:
            # this corresponds to stopping
            # as the level of the lower parent
            # is propagated to the current level
            output[p] = output[q]
        else:
            # here the image reconstruction continues.
            # The level is maintained (original value).
            output[p] = image[p]

    return


# _max_tree is the main function. It allows to construct a max
# tree representation of the image.
cpdef void _max_tree(dtype_t[::1] image,
                     DTYPE_BOOL_t[::1] mask,
                     DTYPE_INT32_t[::1] structure,
                     DTYPE_INT32_t[::1] shape,
                     DTYPE_INT64_t[::1] parent,
                     DTYPE_INT64_t[::1] sorted_indices
                     ):
    """Builds a max-tree.

    Parameters
    ----------

    image : array of arbitrary type
        The flattened image pixels.
    mask : array of int
        An array of the same shape as `image` where each pixel contains a
        nonzero value if it is to be considered for the filtering.
        NOTE: it is *essential* that the border pixels (those
        with neighbors falling outside the volume) are all set to zero, or
        segfaults could occur.
    structure : array of int
        A list of coordinate offsets to compute the raveled coordinates of each
        neighbor from the raveled coordinates of the current pixel.
    parent : array of int
        First output: image of the same shape as the input image. The value
        at each pixel is the parent index of this pixel in the max-tree
        reprentation.
    tree_order : array of int
        Second output: list of length = number of pixels. Each element
        corresponds to one pixel index in the image. It encodes the order
        of elements in the tree: a parent of a pixel always comes before
        the element itself. More formally: i < j implies that j cannot be
        the parent of i.
    """

    cdef DTYPE_UINT64_t number_of_pixels = len(image)
    cdef DTYPE_UINT64_t number_of_dimensions = len(shape)

    cdef DTYPE_INT64_t i = 0
    cdef DTYPE_INT64_t p = 0
    cdef DTYPE_INT64_t root = 0
    cdef DTYPE_INT64_t index = 0

    cdef Py_ssize_t nneighbors = structure.shape[0]

    cdef DTYPE_INT64_t[::1] zpar = parent.copy()

    cdef np.ndarray[DTYPE_INT32_t, ndim=2] points = unravel_offsets(
                                                        structure, shape)

    # initialization of the image parent.
    for i in range(number_of_pixels):
        parent[i] = -1
        zpar[i] = -1

    # traverse the array in reversed order (from highest value to lowest value)
    for p in sorted_indices[::-1]:
        parent[p] = p
        zpar[p] = p

        for i in range(nneighbors):

            # get the ravelled index of the neighbor
            index = p + structure[i]

            if not mask[p]:
                # in this case, p is at the border of the image.
                # some neighbor point is not valid.
                if not _is_valid_neighbor(p, points[i], shape):
                    # neighbor is not in the image.
                    continue

            if parent[index] < 0:
                # in this case the parent is not yet set: we ignore
                continue

            root = find_root(zpar, index)

            if root != p:
                zpar[root] = p
                parent[root] = p

    # In a canonized max-tree, each parent is a canonical pixel,
    # i.e. for each connected component at a level l, all pixels point
    # to the same representative which in turn points to the representative
    # pixel at the next level.
    canonize(image, parent, sorted_indices)

    return
