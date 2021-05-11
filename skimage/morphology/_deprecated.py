from .._shared.utils import deprecated


@deprecated('skimage.segmentation.watershed', removed_version='0.19')
def watershed(image, markers=None, connectivity=1, offset=None, mask=None,
              compactness=0, watershed_line=False):
    """Find watershed basins in `image` flooded from given `markers`.

    Parameters
    ----------
    image : ndarray (2-D, 3-D, ...) of integers
        Data array where the lowest value points are labeled first.
    markers : int, or ndarray of int, same shape as `image`, optional
        The desired number of markers, or an array marking the basins with the
        values to be assigned in the label matrix. Zero means not a marker. If
        ``None`` (no markers given), the local minima of the image are used as
        markers.
    connectivity : ndarray, optional
        An array with the same number of dimensions as `image` whose
        non-zero elements indicate neighbors for connection.
        Following the scipy convention, default is a one-connected array of
        the dimension of the image.
    offset : array_like of shape image.ndim, optional
        offset of the connectivity (one offset per dimension)
    mask : ndarray of bools or 0s and 1s, optional
        Array of same shape as `image`. Only points at which mask == True
        will be labeled.
    compactness : float, optional
        Use compact watershed [3]_ with given compactness parameter.
        Higher values result in more regularly-shaped watershed basins.
    watershed_line : bool, optional
        If watershed_line is True, a one-pixel wide line separates the regions
        obtained by the watershed algorithm. The line has the label 0.

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
    This function implements a watershed algorithm [1]_ [2]_ that apportions
    pixels into marked basins. The algorithm uses a priority queue to hold
    the pixels with the metric for the priority queue being pixel value, then
    the time of entry into the queue - this settles ties in favor of the
    closest marker.
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
    .. [1] https://en.wikipedia.org/wiki/Watershed_%28image_processing%29
    .. [2] http://cmm.ensmp.fr/~beucher/wtshed.html
    .. [3] Peer Neubert & Peter Protzel (2014). Compact Watershed and
           Preemptive SLIC: On Improving Trade-offs of Superpixel Segmentation
           Algorithms. ICPR 2014, pp 996-1001. :DOI:`10.1109/ICPR.2014.181`
           https://www.tu-chemnitz.de/etit/proaut/publications/cws_pSLIC_ICPR.pdf

    Examples
    --------
    The watershed algorithm is useful to separate overlapping objects.

    We first generate an initial image with two overlapping circles:

    >>> import numpy as np
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

    >>> labels = watershed(-distance, markers, mask=image)  # doctest: +SKIP

    The algorithm works also for 3-D images, and can be used for example to
    separate overlapping spheres.
    """
    from ..segmentation import watershed as _watershed
    return _watershed(image, markers, connectivity, offset, mask,
                      compactness, watershed_line)
