"""watershed.py - watershed algorithm

This module implements a watershed algorithm that apportions pixels into
marked basins. The algorithm uses a priority queue to hold the pixels
with the metric for the priority queue being pixel value, then the time
of entry into the queue - this settles ties in favor of the closest marker.

Some ideas taken from
Soille, "Automated Basin Delineation from Digital Elevation Models Using
Mathematical Morphology", Signal Processing 20 (1990) 171-182.

The most important insight in the paper is that entry time onto the queue
solves two problems: a pixel should be assigned to the neighbor with the
largest gradient or, if there is no gradient, pixels on a plateau should
be split between markers on opposite sides.

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
"""

from _heapq import heappush, heappop
import numpy as np
import scipy.ndimage
from ..filter import rank_order

from . import _watershed


def watershed(image, markers, connectivity=None, offset=None, mask=None):
    """
    Return a matrix labeled using the watershed segmentation algorithm

    Parameters
    ----------

    image: ndarray (2-D, 3-D, ...) of integers
        Data array where the lowest value points are labeled first.
    markers: ndarray of the same shape as `image`
        An array marking the basins with the values to be assigned in the
        label matrix. Zero means not a marker. This array should be of an
        integer type.
    connectivity: ndarray, optional
        An array with the same number of dimensions as `image` whose
        non-zero elements indicate neighbors for connection.
        Following the scipy convention, default is a one-connected array of
        the dimension of the image.
    offset: array_like of shape image.ndim, optional
        offset of the connectivity (one offset per dimension)
    mask: ndarray of bools or 0s and 1s, optional
        Array of same shape as `image`. Only points at which mask == True
        will be labeled.

    Returns
    -------
    out: ndarray
        A labeled matrix of the same type and shape as markers

    See also
    --------

    skimage.segmentation.random_walker: random walker segmentation
        A segmentation algorithm based on anisotropic diffusion, usually
        slower than the watershed but with good results on noisy data and
        boundaries with holes.

    Notes
    -----
    This function implements a watershed algorithm [1]_that apportions pixels
    into marked basins. The algorithm uses a priority queue to hold the pixels
    with the metric for the priority queue being pixel value, then the time of
    entry into the queue - this settles ties in favor of the closest marker.

    Some ideas taken from
    Soille, "Automated Basin Delineation from Digital Elevation Models Using
    Mathematical Morphology", Signal Processing 20 (1990) 171-182

    The most important insight in the paper is that entry time onto the queue
    solves two problems: a pixel should be assigned to the neighbor with the
    largest gradient or, if there is no gradient, pixels on a plateau should
    be split between markers on opposite sides.

    This implementation converts all arguments to specific, lowest common
    denominator types, then passes these to a C algorithm.

    Markers can be determined manually, or automatically using for example
    the local minima of the gradient of the image, or the local maxima of the
    distance function to the background for separating overlapping objects
    (see example).

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Watershed_%28image_processing%29

    .. [2] http://cmm.ensmp.fr/~beucher/wtshed.html

    Examples
    --------
    The watershed algorithm is very useful to separate overlapping objects

    >>> # Generate an initial image with two overlapping circles
    >>> x, y = np.indices((80, 80))
    >>> x1, y1, x2, y2 = 28, 28, 44, 52
    >>> r1, r2 = 16, 20
    >>> mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
    >>> mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
    >>> image = np.logical_or(mask_circle1, mask_circle2)
    >>> # Now we want to separate the two objects in image
    >>> # Generate the markers as local maxima of the distance
    >>> # to the background
    >>> from scipy import ndimage
    >>> distance = ndimage.distance_transform_edt(image)
    >>> local_maxi = is_local_maximum(distance, image, np.ones((3, 3)))
    >>> markers = ndimage.label(local_maxi)[0]
    >>> labels = watershed(-distance, markers, mask=image)

    The algorithm works also for 3-D images, and can be used for example to
    separate overlapping spheres.
    """

    if connectivity == None:
        c_connectivity = scipy.ndimage.generate_binary_structure(image.ndim, 1)
    else:
        c_connectivity = np.array(connectivity, bool)
        if c_connectivity.ndim != image.ndim:
            raise ValueError("Connectivity dimension must be same as image")
    if offset == None:
        if any([x % 2 == 0 for x in c_connectivity.shape]):
            raise ValueError("Connectivity array must have an unambiguous "
                    "center")
        #
        # offset to center of connectivity array
        #
        offset = np.array(c_connectivity.shape) // 2

    # pad the image, markers, and mask so that we can use the mask to
    # keep from running off the edges
    pads = offset

    def pad(im):
        new_im = np.zeros([i + 2 * p for i, p in zip(im.shape, pads)], im.dtype)
        new_im[[slice(p, -p, None) for p in pads]] = im
        return new_im

    if mask is not None:
        mask = pad(mask)
    else:
        mask = pad(np.ones(image.shape, bool))
    image = pad(image)
    markers = pad(markers)

    c_image = rank_order(image)[0].astype(np.int32)
    c_markers = np.ascontiguousarray(markers, dtype=np.int32)
    if c_markers.ndim != c_image.ndim:
        raise ValueError("markers (ndim=%d) must have same # of dimensions "
                         "as image (ndim=%d)" % (c_markers.ndim, c_image.ndim))
    if c_markers.shape != c_image.shape:
        raise ValueError("image and markers must have the same shape")
    if mask != None:
        c_mask = np.ascontiguousarray(mask, dtype=bool)
        if c_mask.ndim != c_markers.ndim:
            raise ValueError("mask must have same # of dimensions as image")
        if c_markers.shape != c_mask.shape:
            raise ValueError("mask must have same shape as image")
        c_markers[np.logical_not(mask)] = 0
    else:
        c_mask = None
    c_output = c_markers.copy()

    #
    # We pass a connectivity array that pre-calculates the stride for each
    # neighbor.
    #
    # The result of this bit of code is an array with one row per
    # point to be considered. The first column is the pre-computed stride
    # and the second through last are the x,y...whatever offsets
    # (to do bounds checking).
    c = []
    image_stride = np.array(image.strides) // image.itemsize
    for i in range(np.product(c_connectivity.shape)):
        multiplier = 1
        offs = []
        indexes = []
        ignore = True
        for j in range(len(c_connectivity.shape)):
            idx = (i // multiplier) % c_connectivity.shape[j]
            off = idx - offset[j]
            if off:
                ignore = False
            offs.append(off)
            indexes.append(idx)
            multiplier *= c_connectivity.shape[j]
        if (not ignore) and c_connectivity.__getitem__(tuple(indexes)):
            stride = np.dot(image_stride, np.array(offs))
            offs.insert(0, stride)
            c.append(offs)
    c = np.array(c, np.int32)

    pq, age = __heapify_markers(c_markers, c_image)
    pq = np.ascontiguousarray(pq, dtype=np.int32)
    if np.product(pq.shape) > 0:
        # If nothing is labeled, the output is empty and we don't have to
        # do anything
        c_output = c_output.flatten()
        if c_mask == None:
            c_mask = np.ones(c_image.shape, np.int8).flatten()
        else:
            c_mask = c_mask.astype(np.int8).flatten()
        _watershed.watershed(c_image.flatten(),
                             pq, age, c,
                             c_image.ndim,
                             c_mask,
                             np.array(c_image.shape, np.int32),
                             c_output)
    c_output = c_output.reshape(c_image.shape)[[slice(1, -1, None)] *
                                                image.ndim]
    try:
        return c_output.astype(markers.dtype)
    except:
        return c_output


def is_local_maximum(image, labels=None, footprint=None):
    """
    Return a boolean array of points that are local maxima

    Parameters
    ----------
    image: ndarray (2-D, 3-D, ...)
        intensity image

    labels: ndarray, optional
        find maxima only within labels. Zero is reserved for background.

    footprint: ndarray of bools, optional
        binary mask indicating the neighborhood to be examined
        `footprint` must be a matrix with odd dimensions, the center is taken
        to be the point in question.

    Returns
    -------
    result: ndarray of bools
        mask that is True for pixels that are local maxima of `image`

    Examples
    --------
    >>> image = np.zeros((4, 4))
    >>> image[1, 2] = 2
    >>> image[3, 3] = 1
    >>> image
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.]])
    >>> is_local_maximum(image)
    array([[ True, False, False, False],
           [ True, False,  True, False],
           [ True, False, False, False],
           [ True,  True, False,  True]], dtype=bool)
    >>> image = np.arange(16).reshape((4, 4))
    >>> labels = np.array([[1, 2], [3, 4]])
    >>> labels = np.repeat(np.repeat(labels, 2, axis=0), 2, axis=1)
    >>> labels
    array([[1, 1, 2, 2],
           [1, 1, 2, 2],
           [3, 3, 4, 4],
           [3, 3, 4, 4]])
    >>> image
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> is_local_maximum(image, labels=labels)
    array([[False, False, False, False],
           [False,  True, False,  True],
           [False, False, False, False],
           [False,  True, False,  True]], dtype=bool)
    """
    if labels is None:
        labels = np.ones(image.shape, dtype=np.uint8)
    if footprint is None:
        footprint = np.ones([3] * image.ndim, dtype=np.uint8)
    assert((np.all(footprint.shape) & 1) == 1)
    footprint = (footprint != 0)
    footprint_extent = (np.array(footprint.shape) - 1) // 2
    if np.all(footprint_extent == 0):
        return labels > 0
    result = (labels > 0).copy()
    #
    # Create a labels matrix with zeros at the borders that might be
    # hit by the footprint.
    #
    big_labels = np.zeros(np.array(labels.shape) + footprint_extent * 2,
                          labels.dtype)
    big_labels[[slice(fe, -fe) for fe in footprint_extent]] = labels
    #
    # Find the relative indexes of each footprint element
    #
    image_strides = np.array(image.strides) // image.dtype.itemsize
    big_strides = np.array(big_labels.strides) // big_labels.dtype.itemsize
    result_strides = np.array(result.strides) // result.dtype.itemsize
    footprint_offsets = np.mgrid[[slice(-fe, fe + 1) for fe in footprint_extent]]

    fp_image_offsets = np.sum(image_strides[:, np.newaxis] *
                              footprint_offsets[:, footprint], 0)
    fp_big_offsets = np.sum(big_strides[:, np.newaxis] *
                            footprint_offsets[:, footprint], 0)
    #
    # Get the index of each labeled pixel in the image and big_labels arrays
    #
    indexes = np.mgrid[[slice(0, x) for x in labels.shape]][:, labels > 0]
    image_indexes = np.sum(image_strides[:, np.newaxis] * indexes, 0)
    big_indexes = np.sum(big_strides[:, np.newaxis] *
                         (indexes + footprint_extent[:, np.newaxis]), 0)
    result_indexes = np.sum(result_strides[:, np.newaxis] * indexes, 0)
    #
    # Now operate on the raveled images
    #
    big_labels_raveled = big_labels.ravel()
    image_raveled = image.ravel()
    result_raveled = result.ravel()
    #
    # A hit is a hit if the label at the offset matches the label at the pixel
    # and if the intensity at the pixel is greater or equal to the intensity
    # at the offset.
    #
    for fp_image_offset, fp_big_offset in zip(fp_image_offsets, fp_big_offsets):
        same_label = (big_labels_raveled[big_indexes + fp_big_offset] ==
                      big_labels_raveled[big_indexes])
        less_than = (image_raveled[image_indexes[same_label]] <
                     image_raveled[image_indexes[same_label] + fp_image_offset])
        result_raveled[result_indexes[same_label][less_than]] = False

    return result


# ---------------------- deprecated ------------------------------
# Deprecate slower pure-Python code, that we keep only for
# pedagogical purposes
def __heapify_markers(markers, image):
    """Create a priority queue heap with the markers on it"""
    stride = np.array(image.strides) // image.itemsize
    coords = np.argwhere(markers != 0)
    ncoords = coords.shape[0]
    if ncoords > 0:
        pixels = image[markers != 0]
        age = np.arange(ncoords)
        offset = np.zeros(coords.shape[0], int)
        for i in range(image.ndim):
            offset = offset + stride[i] * coords[:, i]
        pq = np.column_stack((pixels, age, offset, coords))
        # pixels = top priority, age=second
        ordering = np.lexsort((age, pixels))
        pq = pq[ordering, :]
    else:
        pq = np.zeros((0, markers.ndim + 3), int)
    return (pq, ncoords)


def _slow_watershed(image, markers, connectivity=8, mask=None):
    """Return a matrix labeled using the watershed algorithm

    Use the `watershed` function for a faster execution.
    This pure Python function is solely for pedagogical purposes.

    Parameters
    ----------
    image: 2-d ndarray of integers
        a two-dimensional matrix where the lowest value points are
        labeled first.
    markers: 2-d ndarray of integers
        a two-dimensional matrix marking the basins with the values
        to be assigned in the label matrix. Zero means not a marker.
    connectivity: {4, 8}, optional
        either 4 for four-connected or 8 (default) for eight-connected
    mask: 2-d ndarray of bools, optional
        don't label points in the mask

    Returns
    -------
    out: ndarray
        A labeled matrix of the same type and shape as markers


    Notes
    -----
    This function implements a watershed algorithm [1]_that apportions pixels
    into marked basins. The algorithm uses a priority queue to hold the pixels
    with the metric for the priority queue being pixel value, then the time of
    entry into the queue - this settles ties in favor of the closest marker.

    Some ideas taken from
    Soille, "Automated Basin Delineation from Digital Elevation Models Using
    Mathematical Morphology", Signal Processing 20 (1990) 171-182

    The most important insight in the paper is that entry time onto the queue
    solves two problems: a pixel should be assigned to the neighbor with the
    largest gradient or, if there is no gradient, pixels on a plateau should
    be split between markers on opposite sides.

    This implementation converts all arguments to specific, lowest common
    denominator types, then passes these to a C algorithm.

    Markers can be determined manually, or automatically using for example
    the local minima of the gradient of the image, or the local maxima of the
    distance function to the background for separating overlapping objects.
    """
    if connectivity not in (4, 8):
        raise ValueError("Connectivity was %d: it should be either \
        four or eight" % (connectivity))

    image = np.array(image)
    markers = np.array(markers)
    labels = markers.copy()
    max_x = markers.shape[0]
    max_y = markers.shape[1]
    if connectivity == 4:
        connect_increments = ((1, 0), (0, 1), (-1, 0), (0, -1))
    else:
        connect_increments = ((1, 0), (1, 1), (0, 1), (-1, 1),
                              (-1, 0), (-1, -1), (0, -1), (1, -1))
    pq, age = __heapify_markers(markers, image)
    pq = pq.tolist()
    #
    # The second step pops a value off of the queue, then labels and pushes
    # the neighbors
    #
    while len(pq):
        pix_value, pix_age, ignore, pix_x, pix_y = heappop(pq)
        pix_label = labels[pix_x, pix_y]
        for xi, yi in connect_increments:
            x = pix_x + xi
            y = pix_y + yi
            if x < 0 or y < 0 or x >= max_x or y >= max_y:
                continue
            if labels[x, y]:
                continue
            if mask != None and not mask[x, y]:
                continue
            # label the pixel
            labels[x, y] = pix_label
            # put the pixel onto the queue
            heappush(pq, [image[x, y], age, 0, x, y])
            age += 1
    return labels
