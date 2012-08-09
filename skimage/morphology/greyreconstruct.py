"""
`reconstruction` originally part of CellProfiler, code licensed under both GPL and BSD licenses.

Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky

"""
import numpy as np

from skimage.filter.rank_order import rank_order


def reconstruction(image, mask, selem=None, offset=None):
    """Perform a morphological reconstruction of the image.

    Reconstruction requires a "seed" image and a "mask" image. The seed image
    gets dilated until it is constrained by the mask. The "seed" and "mask"
    images will be the minimum and maximum possible values of the reconstructed
    image, respectively.

    Parameters
    ----------
    image : ndarray
        The seed image.
    mask : ndarray
        The maximum allowed value at each point.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.

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
    Vincent, L., "Morphological Grayscale Reconstruction in Image Analysis:
    Applications and Efficient Algorithms", IEEE Transactions on Image
    Processing (1993)

    Examples
    --------
    Uses for greyscale reconstruction are described in Vincent (1993). For
    example, let's try to extract the features of an image by subtracting a
    background image created by reconstruction.

    First, create an image where the "bumps" are the features that
    we want to extract:

    >>> import numpy as np
    >>> from scikits.image.morphology.grey import grey_reconstruction
    >>> y, x = np.mgrid[:20:0.5, :20:0.5]
    >>> bumps = np.sin(x) + np.sin(y)

    To create the background image, set the mask image to the original image,
    and the seed image to the original image with an intensity offset, `h`.

    >>> h = 0.3
    >>> seed = bumps - h
    >>> rec = grey_reconstruction(seed, bumps)

    The resulting reconstructed image looks exactly like the original image,
    but with the peaks of the bumps cut off. Subtracting this reconstructed
    image from the original image leaves just the peaks of the bumps

    >>> hdome = bumps - rec

    This operation is known as the h-dome of the image, which leaves features
    of height `h` in the subtracted image. The h-dome transform, and its
    inverse h-basin, are analogous to the white top-hat and black top-hat
    transforms, but don't require a structuring element.

    """
    assert tuple(image.shape) == tuple(mask.shape)
    assert np.all(image <= mask)
    try:
        from ._greyreconstruct import reconstruction_loop
    except ImportError:
        raise ImportError("_greyreconstruct extension not available.")

    if selem is None:
        selem = np.ones([3]*image.ndim, dtype=bool)
    else:
        selem = selem.copy()

    if offset == None:
        if not all([d % 2 == 1 for d in selem.shape]):
            ValueError("Footprint dimensions must all be odd")
        offset = np.array([d / 2 for d in selem.shape])
    # Cross out the center of the selem
    selem[[slice(d, d + 1) for d in offset]] = False

    # Construct an array that's padded on the edges so we can ignore boundaries
    # The array is a dstack of the image and the mask; this lets us interleave
    # image and mask pixels when sorting which makes list manipulations easier
    padding = (np.array(selem.shape) / 2).astype(int)
    dims = np.zeros(image.ndim + 1, dtype=int)
    dims[1:] = np.array(image.shape) + 2 * padding
    dims[0] = 2
    inside_slices = [slice(p, -p) for p in padding]
    values = np.ones(dims) * np.min(image)
    values[[0] + inside_slices] = image
    values[[1] + inside_slices] = mask

    # Create a list of strides across the array to get the neighbors
    # within a flattened array
    value_stride = np.array(values.strides[1:]) / values.dtype.itemsize
    image_stride = values.strides[0] / values.dtype.itemsize
    selem_mgrid = np.mgrid[[slice(-o, d - o)
                            for d, o in zip(selem.shape, offset)]]
    selem_offsets = selem_mgrid[:, selem].transpose()
    strides = np.array([np.sum(value_stride * selem_offset)
                        for selem_offset in selem_offsets], np.int32)
    values = values.flatten()
    value_sort = np.lexsort([-values]).astype(np.int32)

    # Make a linked list of pixels sorted by value. -1 is the list terminator.
    prev = -np.ones(len(values), np.int32)
    next = -np.ones(len(values), np.int32)
    prev[value_sort[1:]] = value_sort[:-1]
    next[value_sort[:-1]] = value_sort[1:]

    # Create a rank-order value array so that the Cython inner-loop
    # can operate on a uniform data type
    values, value_map = rank_order(values)
    current = value_sort[0]

    reconstruction_loop(values, prev, next, strides, current, image_stride)

    # Reshape the values array to the shape of the padded image
    # and return the unpadded portion of that result
    values = value_map[values[:image_stride]]
    values.shape = np.array(image.shape) + 2 * padding
    return values[inside_slices]

