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

import numpy as np
from scipy import ndimage as ndi

from . import _watershed


def _validate_inputs(image, markers, mask):
    """Ensure that all inputs to watershed have matching shapes and types.

    Parameters
    ----------
    image : array
        The input image.
    markers : array
        The marker image.
    mask : array, or None
        A boolean mask, True where we want to compute the watershed.

    Returns
    -------
    mask : array
        The validated and formatted mask array. If ``None`` was given, it
        is a volume of all ``True`` values.

    Raises
    ------
    ValueError
        If the shapes of the given arrays don't match.
    """
    if markers.shape != image.shape:
        raise ValueError("Markers (shape %s) must have same shape "
                         "as image (shape %s)" % (markers.ndim, image.ndim))
    if mask is not None and mask.shape != image.shape:
        raise ValueError("mask must have same shape as image")
    if mask is None:
        # Use a complete `True` mask if none is provided
        mask = np.ones(image.shape, bool)
    return mask


def _validate_connectivity(image_dim, connectivity, offset):
    """Convert any valid connectivity to a structuring element and offset.

    Parameters
    ----------
    image_dim : int
        The number of dimensions of the input image.
    connectivity : int, array, or None
        The neighborhood connectivity. An integer is interpreted as in
        ``scipy.ndimage.generate_binary_structure``, as the maximum number
        of orthogonal steps to reach a neighbor. An array is directly
        interpreted as a structuring element and its shape is validated against
        the input image shape. ``None`` is interpreted as a connectivity of 1.
    offset : tuple of int, or None
        The coordinates of the center of the structuring element.

    Returns
    -------
    c_connectivity : array of bool
        The structuring element corresponding to the input `connectivity`.
    offset : array of int
        The offset corresponding to the center of the structuring element.

    Raises
    ------
    ValueError:
        If the image dimension and the connectivity or offset dimensions don't
        match.
    """
    if connectivity is None:
        connectivity = 1
    if np.isscalar(connectivity):
        c_connectivity = ndi.generate_binary_structure(image_dim, connectivity)
    else:
        c_connectivity = np.array(connectivity, bool)
        if c_connectivity.ndim != image_dim:
            raise ValueError("Connectivity dimension must be same as image")
    if offset is None:
        if any([x % 2 == 0 for x in c_connectivity.shape]):
            raise ValueError("Connectivity array must have an unambiguous "
                             "center")
        offset = np.array(c_connectivity.shape) // 2
    return c_connectivity, offset


def _compute_neighbors(image, structure, offset):
    #
    # We pass a connectivity array that pre-calculates the stride for each
    # neighbor.
    #
    # The result of this bit of code is an array with one row per
    # point to be considered. The first column is the pre-computed stride
    # and the second through last are the x,y...whatever offsets
    # (to do bounds checking).
    c = []
    distances = []
    image_stride = np.array(image.strides) // image.itemsize
    for i in range(np.product(structure.shape)):
        multiplier = 1
        offs = []
        indexes = []
        ignore = True
        for j in range(len(structure.shape)):
            idx = (i // multiplier) % structure.shape[j]
            off = idx - offset[j]
            if off:
                ignore = False
            offs.append(off)
            indexes.append(idx)
            multiplier *= structure.shape[j]
        if (not ignore) and structure.__getitem__(tuple(indexes)):
            stride = np.dot(image_stride, np.array(offs))
            d = np.sum(np.abs(offs)) - 1
            offs.insert(0, stride)
            c.append(offs)
            distances.append(d)

    c = np.array(c, dtype=np.int32)
    neighborhood = np.ascontiguousarray(c[np.argsort(distances), 0])
    return neighborhood


def watershed(image, markers, connectivity=1, offset=None, mask=None,
              compactness=0):
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
    compactness : float, optional
        Use compact watershed [3]_ with given compactness parameter.

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

    .. [3] Peer Neubert & Peter Protzel (2014). Compact Watershed and
           Preemptive SLIC: On Improving Trade-offs of Superpixel Segmentation
           Algorithms. ICPR 2014, pp 996-1001. DOI:10.1109/ICPR.2014.181
           https://www.tu-chemnitz.de/etit/proaut/forschung/rsrc/cws_pSLIC_ICPR.pdf

    Examples
    --------
    The watershed algorithm is useful to separate overlapping objects.

    We first generate an initial image with two overlapping circles:

    >>> x, y = np.indices((80, 80))
    >>> x1, y1, x2, y2 = 28, 28, 44, 52
    >>> r1, r2 = 16, 20
    >>> mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
    >>> mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
    >>> image = np.logical_or(mask_circle1, mask_circle2)

    Next, we want to separate the two circles. We generate markers at the
    maxima of the distance to the background:

    >>> from scipy import ndimage as ndi
    >>> distance = ndi.distance_transform_edt(image)
    >>> from skimage.feature import peak_local_max
    >>> local_maxi = peak_local_max(distance, labels=image,
    ...                             footprint=np.ones((3, 3)),
    ...                             indices=False)
    >>> markers = ndi.label(local_maxi)[0]

    Finally, we run the watershed on the image and markers:

    >>> labels = watershed(-distance, markers, mask=image)

    The algorithm works also for 3-D images, and can be used for example to
    separate overlapping spheres.
    """
    mask = _validate_inputs(image, markers, mask)
    c_connectivity, offset = _validate_connectivity(image.ndim, connectivity,
                                                    offset)

    # pad the image, markers, and mask so that we can use the mask to
    # keep from running off the edges
    pad_width = [(p, p) for p in offset]
    image = np.pad(image, pad_width, mode='constant')
    mask = np.pad(mask, pad_width, mode='constant')
    markers = np.pad(markers, pad_width, mode='constant')

    c_image = image.astype(np.float64)
    c_mask = np.ascontiguousarray(mask, dtype=np.int8).ravel()
    c_output = np.array(markers, dtype=np.int32).ravel()

    flat_neighborhood = _compute_neighbors(image, c_connectivity, offset)

    marker_locations = np.flatnonzero(markers).astype(np.int32)
    image_strides = np.array(image.strides, dtype=np.int32) // image.itemsize
    if len(marker_locations) > 0:
        _watershed.watershed(c_image.ravel(),
                             marker_locations, flat_neighborhood,
                             c_mask, image_strides, compactness,
                             c_output)
    c_output = c_output.reshape(c_image.shape)[[slice(1, -1, None)] *
                                               image.ndim]
    try:
        return c_output.astype(markers.dtype)
    except:
        return c_output
