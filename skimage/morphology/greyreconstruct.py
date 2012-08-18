"""
This morphological reconstruction routine was adapted from CellProfiler, code
licensed under both GPL and BSD licenses.

Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky

"""
import numpy as np

from skimage.filter.rank_order import rank_order


def reconstruction(seed, mask, selem=None, offset=None, method='dilation'):
    """Perform a morphological reconstruction of an image.

    Reconstruction requires a "seed" image and a "mask" image of equal shape.
    These images set the minimum and maximum possible values of the
    reconstructed image.

    Parameters
    ----------
    seed : ndarray
        The seed image; a.k.a. marker image.
    mask : ndarray
        The maximum allowed value at each point.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    method : {'dilation'|'erosion'}
        Perform reconstruction by dilation or erosion. In dilation (erosion),
        the seed image is dilated (eroded) until limited by the mask image.
        For dilation, each seed value must be less than or equal to the
        corresponding mask value; for erosion, the reverse is true.

    Returns
    -------
    reconstructed : ndarray
       The result of morphological reconstruction.

    Notes
    -----
    The algorithm is taken from:
    Robinson, "Efficient morphological reconstruction: a downhill filter",
    Pattern Recognition Letters 25 (2004) 1759-1767.

    Applications for greyscale reconstruction are discussed in:

    [1] Vincent, L., "Morphological Grayscale Reconstruction in Image Analysis:
        Applications and Efficient Algorithms", IEEE Transactions on Image
        Processing (1993)
    [2] Soille, P., "Morphological Image Analysis: Principles and Applications",
        Chapter 6, 2nd edition (2003), ISBN 3540429883.

    Examples
    --------
    Uses for greyscale reconstruction are described in Vincent (1993). For
    example, let's try to extract the features of an image by subtracting a
    background image created by reconstruction.

    First, create an image where the "bumps" are the features that
    we want to extract:

    >>> import numpy as np
    >>> from skimage.morphology import reconstruction
    >>> y, x = np.mgrid[:20:0.5, :20:0.5]
    >>> bumps = np.sin(x) + np.sin(y)

    To create the background image, set the mask image to the original image,
    and the seed image to the original image with an intensity offset, `h`.

    >>> h = 0.3
    >>> seed = bumps - h
    >>> rec = reconstruction(seed, bumps)

    The resulting reconstructed image looks exactly like the original image,
    but with the peaks of the bumps cut off. Subtracting this reconstructed
    image from the original image leaves just the peaks of the bumps

    >>> hdome = bumps - rec

    This operation is known as the h-dome of the image, which leaves features
    of height `h` in the subtracted image. The h-dome transform, and its
    inverse h-basin, are analogous to the white top-hat and black top-hat
    transforms, but don't require a structuring element.

    """
    assert tuple(seed.shape) == tuple(mask.shape)
    if method == 'dilation' and np.any(seed > mask):
        raise ValueError("Intensity of seed image must be less than that "
                         "of the mask image for reconstruction by dilation.")
    elif method == 'erosion' and np.any(seed < mask):
        raise ValueError("Intensity of seed image must be greater than that "
                         "of the mask image for reconstruction by erosion.")
    try:
        from ._greyreconstruct import reconstruction_loop
    except ImportError:
        raise ImportError("_greyreconstruct extension not available.")

    if selem is None:
        selem = np.ones([3] * seed.ndim, dtype=bool)
    else:
        selem = selem.copy()

    if offset == None:
        if not all([d % 2 == 1 for d in selem.shape]):
            ValueError("Footprint dimensions must all be odd")
        offset = np.array([d / 2 for d in selem.shape])
    # Cross out the center of the selem
    selem[[slice(d, d + 1) for d in offset]] = False

    # Make padding for edges of reconstructed image so we can ignore boundaries
    padding = (np.array(selem.shape) / 2).astype(int)
    dims = np.zeros(seed.ndim + 1, dtype=int)
    dims[1:] = np.array(seed.shape) + 2 * padding
    dims[0] = 2
    inside_slices = [slice(p, -p) for p in padding]
    # Set padded region to minimum image intensity and mask along first axis so
    # we can interleave image and mask pixels when sorting.
    if method == 'dilation':
        pad_value = np.min(seed)
    elif method == 'erosion':
        pad_value = np.max(seed)
    images = np.ones(dims) * pad_value
    images[[0] + inside_slices] = seed
    images[[1] + inside_slices] = mask

    # Create a list of strides across the array to get the neighbors within
    # a flattened array
    value_stride = np.array(images.strides[1:]) / images.dtype.itemsize
    image_stride = images.strides[0] / images.dtype.itemsize
    selem_mgrid = np.mgrid[[slice(-o, d - o)
                            for d, o in zip(selem.shape, offset)]]
    selem_offsets = selem_mgrid[:, selem].transpose()
    nb_strides = np.array([np.sum(value_stride * selem_offset)
                           for selem_offset in selem_offsets], np.int32)

    images = images.flatten()

    # Erosion goes smallest to largest; dilation goes largest to smallest.
    index_sorted = np.argsort(images).astype(np.int32)
    if method == 'dilation':
        index_sorted = index_sorted[::-1]

    # Make a linked list of pixels sorted by value. -1 is the list terminator.
    prev = -np.ones(len(images), np.int32)
    next = -np.ones(len(images), np.int32)
    prev[index_sorted[1:]] = index_sorted[:-1]
    next[index_sorted[:-1]] = index_sorted[1:]

    # Cython inner-loop compares the rank of pixel values.
    if method == 'dilation':
        value_rank, value_map = rank_order(images)
    elif method == 'erosion':
        value_rank, value_map = rank_order(-images)
        value_map = -value_map

    start = index_sorted[0]
    reconstruction_loop(value_rank, prev, next, nb_strides, start, image_stride)

    # Reshape reconstructed image to original image shape and remove padding.
    rec_img = value_map[value_rank[:image_stride]]
    rec_img.shape = np.array(seed.shape) + 2 * padding
    return rec_img[inside_slices]

