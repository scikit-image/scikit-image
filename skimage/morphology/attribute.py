"""attribute.py - apply attribute openings and closings

This module provides functions to apply attribute openings and
closings to arbitrary images. These operators build on the flooding
algorithm used in the watershed transformation, but instead of
partitioning the image plane they stop the flooding once a certain
criterion is met by the region (often termed as 'lake'). The value
of the pixels belonging to this lake are set to the flooding level,
when the flooding stopped.

This implementation provides functions for
1. area openings/closings
2. volume openings/closings
3. diameter openings/closings

References: 
    .. [1] Breen, E.J., Jones, R. (1996). Attribute openings, thinnings and granulometries.
           Computer Vision and Image Understanding 64 (3), 377-389.

Dynamics openings and closings can be implemented by greyreconstruct.
They are therefore not implemented here.
"""
import numpy as np

from .watershed import _validate_connectivity
from .watershed import _compute_neighbors

from .extrema import local_minima
from ..measure import label

from . import _attribute

from ..util import crop


# area_closing and opening are by far the most popular of these operators.
def area_closing(image, area_threshold, mask=None, connectivity=2):
    """Performs an area closing of the image.

    Area closings remove all dark structures of an image with
    a surface smaller than or equal to area_threshold. 
    The output image is thus the smallest image larger than the input
    for which all local minima have at least a surface of 
    area_threshold pixels.

    Area closings are similar to morphological closings, but
    they do not use a fixed structuring element, but rather a deformable
    one, with surface = area_threshold. Consequently, the area_closing
    with area_threshold=1 is the identity.

    Technically, this operator is based on a flooding operation similar
    to the watershed algorithm. But the flooding stops when the criterion
    is fulfilled (here: surface of the region >= area_threshold).

    Parameters
    ----------
    img: ndarray
        The input image for which the area_closing is to be calculated.
        This image can be of any type.
    area_threshold: unsigned int
        The size parameter (number of pixels).
    mask: ndarray of bools or 0s and 1s, optional
        Array of same shape as `image`. Only points at which mask == True
        will be labeled.
    connectivity: unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. It is 1 for 4-connectivity
        and 2 for 8-connectivity. Default value is 1.

    Returns
    -------
    output: ndarray
        area closing of img, which is an image of the same type as img,
        larger than or equal to img for each pixel, with local minima
        with a size of at least area_threshold.

    See also
    --------
    skimage.morphology.extrema.h_minima
    skimage.morphology.extrema.h_maxima
    skimage.morphology.attribute.area_opening
    skimage.morphology.attribute.diameter_opening
    skimage.morphology.attribute.diameter_closing
    skimage.morphology.attribute.volume_fill


    References
    ----------
    .. [1] Vincent L., Proc. "Grayscale area openings and closings,
           their efficient implementation and applications",
           EURASIP Workshop on Mathematical Morphology and its
           Applications to Signal Processing, Barcelona, Spain, pp.22-27, May 1993.
    .. [2] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import attribute

    We create an image (quadratic function with a minimum in the center and
    4 additional local minima.

    >>> w = 12
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:3,1:5] = 160; f[2:4,9:11] = 140; f[9:11,2:4] = 120; f[9:10,9:11] = 100; f[10,10] = 100
    >>> f = f.astype(np.int)

    We can calculate the area closing:

    >>> closed = attribute.area_closing(f, 8, connectivity=1)

    The small (but deep) minima are removed.
    """
    if mask is not None and mask.shape != image.shape:
        raise ValueError("mask must have same shape as image")
    if mask is None:
        # Use a complete `True` mask if none is provided
        mask = np.ones(image.shape, bool)

    neighbors, offset = _validate_connectivity(image.ndim, connectivity,
                                               offset=None)

    seeds_bin = local_minima(image, selem = neighbors)
    seeds = label(seeds_bin, connectivity = connectivity).astype(np.uint64)
    output = image.copy()

    image = np.pad(image, 1, mode='constant')
    mask = np.pad(mask, 1, mode='constant')
    seeds = np.pad(seeds, 1, mode='constant')
    output = np.pad(output, 1, mode='constant')

    flat_neighborhood = _compute_neighbors(image, neighbors, offset)
    image_strides = np.array(image.strides, dtype=np.int32) // image.itemsize

    _attribute.area_closing(image.ravel(),
                           area_threshold,
                           seeds.ravel(),
                           flat_neighborhood,
                           mask.ravel().astype(np.uint8),
                           image_strides,
                           0.000001,
                           output.ravel()
                           )
    output = crop(output, 1, copy=True)
    return(output)


# area_closing and opening are by far the most popular of these operators.
def area_opening(image, area_threshold, mask=None, connectivity=2):
    """Performs an area opening of the image.

    Area opening remove all bright structures of an image with
    a surface smaller than or equal to area_threshold. 
    The output image is thus the largest image smaller than the input
    for which all local maxima have at least a surface of 
    area_threshold pixels.

    Area openings are similar to morphological openings, but
    they do not use a fixed structuring element, but rather a deformable
    one, with surface = area_threshold. Consequently, the area_opening
    with area_threshold=1 is the identity.

    Technically, this operator is based on a flooding operation similar
    to the watershed algorithm. But the flooding stops when the criterion
    is fulfilled (here: surface of the region >= area_threshold).

    Parameters
    ----------
    img: ndarray
        The input image for which the area_opening is to be calculated.
        This image can be of any type.
    area_threshold: unsigned int
        The size parameter (number of pixels).
    mask: ndarray of bools or 0s and 1s, optional
        Array of same shape as `image`. Only points at which mask == True
        will be labeled.
    connectivity: unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. It is 1 for 4-connectivity
        and 2 for 8-connectivity. Default value is 1.

    Returns
    -------
    output: ndarray
        area opening of img, which is an image of the same type as img,
        smaller than or equal to img for each pixel, with local maxima
        with a size of at least area_threshold.

    See also
    --------
    skimage.morphology.attribute.area_closing
    skimage.morphology.extrema.h_minima
    skimage.morphology.extrema.h_maxima
    skimage.morphology.attribute.diameter_opening
    skimage.morphology.attribute.diameter_closing
    skimage.morphology.attribute.volume_fill


    References
    ----------
    .. [1] Vincent L., Proc. "Grayscale area openings and closings,
           their efficient implementation and applications",
           EURASIP Workshop on Mathematical Morphology and its
           Applications to Signal Processing, Barcelona, Spain, pp.22-27, May 1993.
    .. [2] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import attribute

    We create an image (quadratic function with a maximum in the center and
    4 additional local maxima.

    >>> w = 12
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 20 - 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:3,1:5] = 40; f[2:4,9:11] = 60; f[9:11,2:4] = 80; f[9:10,9:11] = 100; f[10,10] = 100
    >>> f = f.astype(np.int)

    We can calculate the area opening:

    >>> open = attribute.area_opening(f, 8, connectivity=1)

    The small (but high) maxima are removed.
    """
    # the area opening is implemented as the dual operator of area_closing.
    maxval = img.max()
    temp_img = maxval - img
    temp_op = area_closing(temp_img, area_threshold, mask=None, connectivity=2)
    result = maxval - temp_op
    return(result)


def diameter_closing(image, diameter_threshold, mask=None, connectivity=2):
    """Performs a diameter closing of the image.

    Diameter closings remove all dark structures of an image with
    a diameter smaller than or equal to diameter_threshold. 
    The output image is thus the smallest image larger than the input
    for which all local minima have at least a diameter of 
    diameter_threshold pixels.

    Here, the diameter of a region is defined as its maximal extension.
    The operator is also called bounding box closing, as it removes all
    dark structures that would be inside of at least one bounding box of 
    maximal extension diameter_threshold.

    Diameter closings are similar to morphological closings, but
    they do not use a fixed structuring element, but rather a deformable
    one, with maximal extension = diameter_threshold. Consequently, the 
    diameter_closing with diameter_threshold=1 is the identity.
    This operator is particularly useful in distinguishing small objects
    from long, thin objects, which is not so easy with traditional morphological
    closings.

    Technically, this operator is based on a flooding operation similar
    to the watershed algorithm. But the flooding stops when the criterion
    is fulfilled (here: diameter of the region >= diameter_threshold).

    Parameters
    ----------
    img: ndarray
        The input image for which the area_closing is to be calculated.
        This image can be of any type.
    diameter_threshold: unsigned int
        The size parameter (maximal extension in number of pixels).
    mask: ndarray of bools or 0s and 1s, optional
        Array of same shape as `image`. Only points at which mask == True
        will be labeled.
    connectivity: unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. It is 1 for 4-connectivity
        and 2 for 8-connectivity. Default value is 1.

    Returns
    -------
    output: ndarray
        diameter closing of img, which is an image of the same type as img,
        larger than or equal to img for each pixel, with local minima
        with a maximal extension of at least diameter_threshold.

    See also
    --------
    skimage.morphology.attribute.diameter_opening
    skimage.morphology.attribute.area_opening
    skimage.morphology.attribute.area_closing
    skimage.morphology.extrema.h_minima
    skimage.morphology.extrema.h_maxima
    skimage.morphology.attribute.volume_fill


    References
    ----------
    .. [1] Walter, T., & Klein, J.-C. (2002). Automatic Detection of Microaneurysms in 
           Color Fundus images of the human retina by means of the bounding box closing.
           International Symposium on Medical Data Analysis ISMDA (pp. 210-220).
    .. [2] Breen, E.J., Jones, R. (1996). Attribute openings, thinnings and granulometries.
           Computer Vision and Image Understanding 64 (3), 377-389.


    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import attribute

    We create an image (quadratic function with a minimum in the center and
    4 additional local minima.

    >>> w = 12
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:3,1:5] = 160; f[2:4,9:11] = 140; f[9:11,2:4] = 120; f[9:10,9:11] = 100; f[10,10] = 100
    >>> f = f.astype(np.int)

    We can calculate the diameter closing:

    >>> closed = attribute.diameter_closing(f, 3, connectivity=1)

    The small (but deep) dark objects are removed, except for the longest one.
    """

    if mask is not None and mask.shape != image.shape:
        raise ValueError("mask must have same shape as image")
    if mask is None:
        # Use a complete `True` mask if none is provided
        mask = np.ones(image.shape, bool)

    neighbors, offset = _validate_connectivity(image.ndim, connectivity,
                                                  offset=None)

    seeds_bin = local_minima(image, selem = neighbors)
    seeds = label(seeds_bin, connectivity = connectivity).astype(np.uint64)
    output = image.copy()

    image = np.pad(image, 1, mode='constant')
    mask = np.pad(mask, 1, mode='constant')
    seeds = np.pad(seeds, 1, mode='constant')
    output = np.pad(output, 1, mode='constant')

    flat_neighborhood = _compute_neighbors(image, neighbors, offset)
    image_strides = np.array(image.strides, dtype=np.int32) // image.itemsize

    _attribute.diameter_closing(image.ravel(),
                               diameter_threshold,
                               seeds.ravel(),
                               flat_neighborhood,
                               mask.ravel().astype(np.uint8),
                               image_strides,
                               0.000001,
                               output.ravel()
                               )
    output = crop(output, 1, copy=True)

    return(output)


def diameter_opening(image, diameter_threshold, mask=None, connectivity=2):
    """Performs a diameter opening of the image.

    Diameter openings remove all dark structures of an image with
    a diameter smaller than or equal to diameter_threshold. 
    The output image is thus the largest image smaller than the input
    for which all local maxima have at least a diameter of 
    diameter_threshold pixels.

    Here, the diameter of a region is defined as its maximal extension.
    The operator is also called bounding box closing, as it removes all
    bright structures that would be inside of at least one bounding box of 
    maximal extension diameter_threshold.

    Diameter openings are similar to morphological openings, but
    they do not use a fixed structuring element, but rather a deformable
    one, with maximal extension = diameter_threshold. Consequently, the 
    diameter_opening with diameter_threshold=1 is the identity.
    This operator is particularly useful in distinguishing small objects
    from long, thin objects.

    Technically, this operator is based on a flooding operation similar
    to the watershed algorithm. But the flooding stops when the criterion
    is fulfilled (here: diameter of the region >= diameter_threshold).

    Parameters
    ----------
    img: ndarray
        The input image for which the area_closing is to be calculated.
        This image can be of any type.
    diameter_threshold: unsigned int
        The size parameter (maximal extension in number of pixels).
    mask: ndarray of bools or 0s and 1s, optional
        Array of same shape as `image`. Only points at which mask == True
        will be labeled.
    connectivity: unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. It is 1 for 4-connectivity
        and 2 for 8-connectivity. Default value is 1.

    Returns
    -------
    output: ndarray
        diameter opening of img, which is an image of the same type as img,
        smaller than or equal to img for each pixel, with local maxima
        with a maximal extension of at least diameter_threshold.

    See also
    --------
    skimage.morphology.attribute.diameter_closing
    skimage.morphology.attribute.area_opening
    skimage.morphology.attribute.area_closing
    skimage.morphology.extrema.h_minima
    skimage.morphology.extrema.h_maxima
    skimage.morphology.attribute.volume_fill


    References
    ----------
    .. [1] Walter, T., & Klein, J.-C. (2002). Automatic Detection of Microaneurysms in 
           Color Fundus images of the human retina by means of the bounding box closing.
           International Symposium on Medical Data Analysis ISMDA (pp. 210-220).
    .. [2] Breen, E.J., Jones, R. (1996). Attribute openings, thinnings and granulometries.
           Computer Vision and Image Understanding 64 (3), 377-389.


    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import attribute

    We create an image (quadratic function with a maximum in the center and
    4 additional local maxima.

    >>> w = 12
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 20 - 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:3,1:5] = 40; f[2:4,9:11] = 60; f[9:11,2:4] = 80; f[9:10,9:11] = 100; f[10,10] = 100
    >>> f = f.astype(np.int)

    We can calculate the diameter opening:

    >>> closed = attribute.diameter_opening(f, 3, connectivity=1)

    The small bright objects are removed, except for the longest one.
    The larger bright object in the middle remains also untouched.
    """
    # the diameter opening is implemented as the dual operator of diameter_closing.
    maxval = img.max()
    temp_img = maxval - img
    temp_op = diameter_closing(temp_img, area_threshold, mask=None, connectivity=2)
    result = maxval - temp_op
    return(result)


def volume_fill(image, volume_threshold, mask=None, connectivity=2):
    """Performs a volume filling of the image.

    This function corresponds to a flooding of the image where - loosely
    speaking - each local minimum is flooded with the same volume of liquid.
    The local minima in the input image are either completely filled if their
    volume is smaller than volume_threshold or increased at a level such
    that the volume (sum of grey levels) added to the catchment basin
    is at least volume_threshold.

    In contrast to area closings or diameter closings, this operator is not a
    closing, as repeating the operator will continue modifying the image
    (and hence, it is not idempotent).

    The idea of this operator is that it provides a compromise between
    the widely used area closings and h-minima.

    Technically, this operator is based on a flooding operation similar
    to the watershed algorithm. But the flooding stops when the criterion
    is fulfilled (here: volume of the lake >= volume_threshold).

    Parameters
    ----------
    img: ndarray
        The input image for which the area_closing is to be calculated.
        This image can be of any type.
    volume_threshold: unsigned int
        The integral of grey levels added to each local minimum.
    mask: ndarray of bools or 0s and 1s, optional
        Array of same shape as `image`. Only points at which mask == True
        will be labeled.
    connectivity: unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. It is 1 for 4-connectivity
        and 2 for 8-connectivity. Default value is 1.

    Returns
    -------
    output: ndarray
        volume fill of img, which is an image of the same type as img,
        each local minimum filled by the same signal amount.

    See also
    --------
    skimage.morphology.attribute.area_opening
    skimage.morphology.attribute.area_closing
    skimage.morphology.extrema.h_minima
    skimage.morphology.extrema.h_maxima
    skimage.morphology.attribute.diameter_closing
    skimage.morphology.attribute.diameter_opening


    References
    ----------
    .. [1] Vachier, C., Meyer, F. (1995). Extinction values: a new measurement of persistence.
           In: Proceedings of the IEEE Workshop on Non Linear Signal and Image Processing (pp. 254-257).


    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import attribute

    We create an image (quadratic function with a minimum in the center and
    4 additional local minima.

    >>> w = 12
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:3,1:5] = 160; f[2:4,9:11] = 140; f[9:11,2:4] = 120; f[9:10,9:11] = 100; f[10,10] = 100
    >>> f = f.astype(np.int)

    We can calculate the diameter closing:

    >>> filled = attribute.volume_fill(f, 30, connectivity=1)

    All minima (large minima and small minima) will be filled by the same amount.
    """
    if mask is not None and mask.shape != image.shape:
        raise ValueError("mask must have same shape as image")
    if mask is None:
        # Use a complete `True` mask if none is provided
        mask = np.ones(image.shape, bool)

    neighbors, offset = _validate_connectivity(image.ndim, connectivity,
                                                  offset=None)

    seeds_bin = local_minima(image, selem = neighbors)
    seeds = label(seeds_bin, connectivity = connectivity).astype(np.uint64)
    output = image.copy()

    image = np.pad(image, 1, mode='constant')
    mask = np.pad(mask, 1, mode='constant')
    seeds = np.pad(seeds, 1, mode='constant')
    output = np.pad(output, 1, mode='constant')

    flat_neighborhood = _compute_neighbors(image, neighbors, offset)
    image_strides = np.array(image.strides, dtype=np.int32) // image.itemsize

    _attribute.volume_fill(image.ravel(),
                          volume_threshold,
                          seeds.ravel(),
                          flat_neighborhood,
                          mask.ravel().astype(np.uint8),
                          image_strides,
                          0.000001,
                          output.ravel()
                          )
    output = crop(output, 1, copy=True)

    return(output)


