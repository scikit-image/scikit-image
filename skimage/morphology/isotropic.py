"""
Binary morphological operations
"""
import numpy as np
from scipy import ndimage as ndi


def isotropic_erosion(image, radius, out=None, spacing=None):
    """Return binary morphological erosion of an image.

    This function returns the same result as binary erosion but performs
    faster for large circular structuring elements.
    This works by applying a threshold to the exact Euclidean distance map
    of the image. [1]_, [2]_

    Morphological erosion sets a pixel at ``(i,j)`` to the minimum over all
    pixels in the neighborhood centered at ``(i,j)``. Erosion shrinks bright
    regions and enlarges dark regions.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius by which regions should be eroded.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    spacing : float, or sequence of float, optional
        Spacing of elements along each dimension. If a sequence, must be of
        length equal to the input rank; if a single number, this is used for
        all axes. If not specified, a grid spacing of unity is implied.

    Returns
    -------
    eroded : ndarray of bool
        The result of the morphological erosion taking values in
        ``[False, True]``.

    References
    ---------------
    .. [1] Cuisenaire, O. and Macq, B., "Fast Euclidean morphological operators
        using local distance transformation by propagation, and applications,"
        Image Processing And Its Applications, 1999. Seventh International
        Conference on (Conf. Publ. No. 465), 1999, pp. 856-860 vol.2.
        :DOI:`10.1049/cp:19990446`

    .. [2] Ingemar Ragnemalm, Fast erosion and dilation by contour processing
        and thresholding of distance maps, Pattern Recognition Letters,
        Volume 13, Issue 3, 1992, Pages 161-166.
        :DOI:`10.1016/0167-8655(92)90055-5`
    """

    dist = ndi.distance_transform_edt(image, sampling=spacing)
    return np.greater(dist, radius, out=out)


def isotropic_dilation(image, radius, out=None, spacing=None):
    """Return binary morphological dilation of an image.

    This function returns the same result as binary dilation but performs
    faster for large circular structuring elements.
    This works by applying a threshold to the exact Euclidean distance map
    of the inverted image. [1]_, [2]_

    Morphological dilation sets a pixel at ``(i,j)`` to the maximum over all
    pixels in the neighborhood centered at ``(i,j)``. Dilation enlarges bright
    regions and shrinks dark regions.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius by which regions should be dilated.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    spacing : float, or sequence of float, optional
        Spacing of elements along each dimension. If a sequence, must be of
        length equal to the input rank; if a single number, this is used for
        all axes. If not specified, a grid spacing of unity is implied.

    Returns
    -------
    dilated : ndarray of bool
        The result of the morphological dilation with values in
        ``[False, True]``.

    References
    ---------------
    .. [1] Cuisenaire, O. and Macq, B., "Fast Euclidean morphological operators
        using local distance transformation by propagation, and applications,"
        Image Processing And Its Applications, 1999. Seventh International
        Conference on (Conf. Publ. No. 465), 1999, pp. 856-860 vol.2.
        :DOI:`10.1049/cp:19990446`

    .. [2] Ingemar Ragnemalm, Fast erosion and dilation by contour processing
        and thresholding of distance maps, Pattern Recognition Letters,
        Volume 13, Issue 3, 1992, Pages 161-166.
        :DOI:`10.1016/0167-8655(92)90055-5`
    """

    dist = ndi.distance_transform_edt(image == 0, sampling=spacing)
    return np.less_equal(dist, radius, out=out)


def isotropic_opening(image, radius, out=None, spacing=None):
    """Return binary morphological opening of an image.

    This function returns the same result as binary opening but performs
    faster for large circular structuring elements.
    This works by thresholding the exact Euclidean distance map. [1]_, [2]_

    The morphological opening on an image is defined as an erosion followed by
    a dilation. Opening can remove small bright spots (i.e. "salt") and connect
    small dark cracks. This tends to "open" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius with which the regions should be opened.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None
        is passed, a new array will be allocated.
    spacing : float, or sequence of float, optional
        Spacing of elements along each dimension. If a sequence, must be of
        length equal to the input rank; if a single number, this is used for
        all axes. If not specified, a grid spacing of unity is implied.

    Returns
    -------
    opened : ndarray of bool
        The result of the morphological opening.

    References
    ---------------
    .. [1] Cuisenaire, O. and Macq, B., "Fast Euclidean morphological operators
        using local distance transformation by propagation, and applications,"
        Image Processing And Its Applications, 1999. Seventh International
        Conference on (Conf. Publ. No. 465), 1999, pp. 856-860 vol.2.
        :DOI:`10.1049/cp:19990446`

    .. [2] Ingemar Ragnemalm, Fast erosion and dilation by contour processing
        and thresholding of distance maps, Pattern Recognition Letters,
        Volume 13, Issue 3, 1992, Pages 161-166.
        :DOI:`10.1016/0167-8655(92)90055-5`
    """

    eroded = isotropic_erosion(image, radius, out=out, spacing=spacing)
    return isotropic_dilation(eroded, radius, out=out, spacing=spacing)


def isotropic_closing(image, radius, out=None, spacing=None):
    """Return binary morphological closing of an image.

    This function returns the same result as binary closing but performs
    faster for large circular structuring elements.
    This works by thresholding the exact Euclidean distance map. [1]_, [2]_

    The morphological closing on an image is defined as a dilation followed by
    an erosion. Closing can remove small dark spots (i.e. "pepper") and connect
    small bright cracks. This tends to "close" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius with which the regions should be closed.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None,
        is passed, a new array will be allocated.
    spacing : float, or sequence of float, optional
        Spacing of elements along each dimension. If a sequence, must be of
        length equal to the input rank; if a single number, this is used for
        all axes. If not specified, a grid spacing of unity is implied.

    Returns
    -------
    closed : ndarray of bool
        The result of the morphological closing.

    References
    ---------------
    .. [1] Cuisenaire, O. and Macq, B., "Fast Euclidean morphological operators
        using local distance transformation by propagation, and applications,"
        Image Processing And Its Applications, 1999. Seventh International
        Conference on (Conf. Publ. No. 465), 1999, pp. 856-860 vol.2.
        :DOI:`10.1049/cp:19990446`

    .. [2] Ingemar Ragnemalm, Fast erosion and dilation by contour processing
        and thresholding of distance maps, Pattern Recognition Letters,
        Volume 13, Issue 3, 1992, Pages 161-166.
        :DOI:`10.1016/0167-8655(92)90055-5`
    """

    dilated = isotropic_dilation(image, radius, out=out, spacing=spacing)
    return isotropic_erosion(dilated, radius, out=out, spacing=spacing)
