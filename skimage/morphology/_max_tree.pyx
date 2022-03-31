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
cimport numpy as np
cimport cython
from .._shared.fused_numerics cimport np_real_numeric

np.import_array()

ctypedef np.float64_t DTYPE_FLOAT64_t
ctypedef np.int32_t DTYPE_INT32_t
ctypedef np.uint32_t DTYPE_UINT32_t
ctypedef np.uint64_t DTYPE_UINT64_t
ctypedef np.int64_t DTYPE_INT64_t
ctypedef np.uint8_t DTYPE_BOOL_t
ctypedef np.uint8_t DTYPE_UINT8_t


cdef DTYPE_INT64_t find_root_rec(DTYPE_INT64_t[::1] parent,
                                 DTYPE_INT64_t index):
    """Get the root of the current tree through a recursive algorithm.

    This function modifies the tree in-place through path compression, which
    reduces the complexity from O(n*n) to O(n*log(n)). Despite path
    compression, our tests showed that the non-recursive version
    (:func:`find_root`) seems to perform better. We leave this version as
    inspiration for future improvements.

    Parameters
    ----------
    parent : array of int
        The array containing parent relationships.
    index : int
        The index of which we want to find the root.

    Returns
    -------
    root : int
        The root found from ``index``.
    """
    if parent[index] != index:
        parent[index] = find_root_rec(parent, parent[index])
    return parent[index]


cdef inline DTYPE_INT64_t find_root(DTYPE_INT64_t[::1] parent,
                                    DTYPE_INT64_t index):
    """Get the root of the current tree.

    Here, we do without path compression and accept the higher complexity, but
    the function is inline and avoids some overhead induced by its recursive
    version.

    Parameters
    ----------
    parent : array of int
        The array containing parent relationships.
    index : int
        The index of which we want to find the root.

    Returns
    -------
    root : int
        The root found from ``index``.
    """
    while parent[index] != parent[parent[index]]:
        parent[index] = parent[parent[index]]
    return parent[index]


cdef void canonize(np_real_numeric[::1] image, DTYPE_INT64_t[::1] parent,
                   DTYPE_INT64_t[::1] sorted_indices):
    """Generate a max-tree for which every node's parent is a canonical node.

    The parent of a non-canonical pixel is a canonical pixel.
    The parent of a canonical pixel is also a canonical pixel with a different
    value. There is exactly one canonical pixel for each component in the
    component tree.

    Parameters
    ----------
    image : array
        The raveled image intensity values.
    parent : array of int
        The array mapping image indices to their parents in the max-tree.
        **This array will be modified in-place.**
    sorted_indices : array of int
        Array of image indices such that if i comes before j, then i cannot
        be the parent of j.
    """
    cdef DTYPE_INT64_t q = 0
    cdef DTYPE_INT64_t p
    for p in sorted_indices:
        q = parent[p]
        if image[q] == image[parent[q]]:
            parent[p] = parent[q]


cdef np.ndarray[DTYPE_INT32_t, ndim = 2] unravel_offsets(
        DTYPE_INT32_t[::1] offsets,
        DTYPE_INT32_t[::1] center_point,
        DTYPE_INT32_t[::1] shape):
    """Unravel a list of offset indices.

    These offsets can be negative. The function generates an array of shape
    (number of offsets, image dimensions), where each row corresponds
    to the coordinates of each point.

    See also
    --------
    unravel_index
    """

    cdef DTYPE_INT32_t number_of_dimensions = len(shape)
    cdef DTYPE_INT32_t number_of_points = len(offsets)
    cdef np.ndarray[DTYPE_INT32_t, ndim = 2] points = np.zeros(
                                                        (number_of_points,
                                                         number_of_dimensions),
                                                        dtype=np.int32)
    cdef DTYPE_INT32_t neg_shift = np.ravel_multi_index(center_point, shape)

    cdef DTYPE_INT32_t i, offset, curr_index, coord

    for i, offset in enumerate(offsets):
        current_point = np.unravel_index(offset + neg_shift, shape)
        for d in range(number_of_dimensions):
            points[i, d] = current_point[d] - center_point[d]

    return points


cdef DTYPE_UINT8_t _is_valid_neighbor(DTYPE_INT64_t index,
                                      DTYPE_INT32_t[::1] coordinates,
                                      DTYPE_INT32_t[::1] shape):
    """Check whether a neighbor of a given pixel is inside the image.

    Parameters
    ----------
    index : int
        The pixel given as a linear index into the raveled image array.
    coordinates : array of int, shape ``image.ndim``
        The neighbor given as a list of offsets from `pixel` in each dimension.
    shape : array of int, shape ``image.ndim`
        The image shape.

    Returns
    -------
    is_neighbor : uint8
        0 if the neighbor falls outside the image, 1 otherwise.
    """

    cdef DTYPE_INT64_t number_of_dimensions = len(shape)
    cdef DTYPE_INT64_t res_coord = 0
    cdef int i = 0

    cdef np.ndarray[DTYPE_INT32_t, ndim = 1] p_coord = np.array(
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


cpdef np.ndarray[DTYPE_FLOAT64_t, ndim = 1] _compute_area(np_real_numeric[::1] image,
            DTYPE_INT64_t[::1] parent,
            DTYPE_INT64_t[::1] sorted_indices):
    """Compute the area of all max-tree components.

    This attribute is used for area opening and closing
    """
    cdef DTYPE_INT64_t p_root = sorted_indices[0]
    cdef DTYPE_INT64_t p, q
    cdef DTYPE_UINT64_t number_of_pixels = len(image)
    cdef np.ndarray[DTYPE_FLOAT64_t, ndim = 1] area = np.ones(number_of_pixels,
                                                              dtype=np.float64)

    for p in sorted_indices[::-1]:
        if p == p_root:
            continue
        q = parent[p]
        area[q] = area[q] + area[p]

    return area


cpdef np.ndarray[DTYPE_FLOAT64_t, ndim = 1] _compute_extension(
            np_real_numeric[::1] image,
            DTYPE_INT32_t[::1] shape,
            DTYPE_INT64_t[::1] parent,
            DTYPE_INT64_t[::1] sorted_indices):
    """Compute the bounding box extension of all max-tree components.

    This attribute is used for diameter opening and closing.
    """
    cdef DTYPE_INT64_t p_root = sorted_indices[0]
    cdef DTYPE_INT64_t p, q
    cdef DTYPE_UINT64_t number_of_pixels = len(image)
    cdef np.ndarray[DTYPE_FLOAT64_t, ndim = 1] extension = np.ones(
                        number_of_pixels,
                        dtype=np.float64)
    cdef np.ndarray[DTYPE_FLOAT64_t, ndim = 2] max_coord = np.array(
                        np.unravel_index(np.arange(number_of_pixels), shape),
                        dtype=np.float64).T
    cdef np.ndarray[DTYPE_FLOAT64_t, ndim = 2] min_coord = np.array(
                        np.unravel_index(np.arange(number_of_pixels), shape),
                        dtype=np.float64).T

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
cpdef void _max_tree_local_maxima(np_real_numeric[::1] image,
                                  DTYPE_UINT64_t[::1] output,
                                  DTYPE_INT64_t[::1] parent,
                                  DTYPE_INT64_t[::1] sorted_indices
                                  ):
    """Find the local maxima in image from the max-tree representation.

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
cpdef void _direct_filter(np_real_numeric[::1] image,
                          np_real_numeric[::1] output,
                          DTYPE_INT64_t[::1] parent,
                          DTYPE_INT64_t[::1] sorted_indices,
                          DTYPE_FLOAT64_t[::1] attribute,
                          DTYPE_FLOAT64_t attribute_threshold
                          ):
    """Apply a direct filtering.

    This produces an image in which for all possible thresholds, each connected
    component has the specified attribute value greater than that threshold.
    This is the basic function called by :func:`area_opening`,
    :func:`diameter_opening`, and similar.

    For :func:`area_opening`, for instance, the attribute is the area.  In this
    case, an image is produced for which all connected components for all
    thresholds have at least an area (pixel count) of the threshold given by
    the user.

    Parameters
    ----------

    image : array
        The flattened image pixels.
    output : array, same size and type as `image`
        The array into which to write the output values. **This array will be
        modified in-place.**
    parent : array of int, same shape as `image`
        Image of indices. The value at each pixel is the index of this pixel's
        parent in the max-tree reprentation.
    sorted_indices : array of int, same shape as `image`
        "List" of pixel indices, which contains an ordering of elements in the
        tree such that a parent of a pixel always comes before the element
        itself. More formally: i < j implies that j cannot be the parent of i.
    attribute : array of float
        Contains the attributes computed for the max-tree.
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
cpdef void _max_tree(np_real_numeric[::1] image,
                     DTYPE_BOOL_t[::1] mask,
                     DTYPE_INT32_t[::1] structure,
                     DTYPE_INT32_t[::1] offset,
                     DTYPE_INT32_t[::1] shape,
                     DTYPE_INT64_t[::1] parent,
                     DTYPE_INT64_t[::1] sorted_indices
                     ):
    """Build a max-tree.

    Parameters
    ----------
    image : array
        The flattened image pixels.
    mask : array of int
        An array of the same shape as `image` where each pixel contains a
        nonzero value if it is to be considered for the filtering.  NOTE: it is
        *essential* that the border pixels (those with neighbors falling
        outside the volume) are all set to zero, or segfaults could occur.
    structure : array of int
        A list of coordinate offsets to compute the raveled coordinates of each
        neighbor from the raveled coordinates of the current pixel.
    parent : array of int
        Output image of the same shape as the input image. The value at each
        pixel is the parent index of this pixel in the max-tree reprentation.
        **This array will be written to in-place.**
    sorted_indices : array of int
        Output "list" of pixel indices, which contains an ordering of elements
        in the tree such that a parent of a pixel always comes before the
        element itself. More formally: i < j implies that j cannot be the
        the parent of i. **This array will be written to in-place.**
    """

    cdef DTYPE_UINT64_t number_of_pixels = len(image)
    cdef DTYPE_UINT64_t number_of_dimensions = len(shape)

    cdef DTYPE_INT64_t i = 0
    cdef DTYPE_INT64_t p = 0
    cdef DTYPE_INT64_t root = 0
    cdef DTYPE_INT64_t index = 0

    cdef Py_ssize_t nneighbors = structure.shape[0]

    cdef DTYPE_INT64_t[::1] zpar = parent.copy()

    cdef np.ndarray[DTYPE_INT32_t, ndim = 2] points = unravel_offsets(
            structure, offset, shape)

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
