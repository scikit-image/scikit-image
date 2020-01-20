from warnings import warn


def watershed(image, markers=None, connectivity=1, offset=None, mask=None,
              compactness=0, watershed_line=False):
    """Find watershed basins in `image` flooded from given `markers`.

    Parameters
    ----------
    image: ndarray (2-D, 3-D, ...) of integers
        Data array where the lowest value points are labeled first.
    markers: int, or ndarray of int, same shape as `image`, optional
        The desired number of markers, or an array marking the basins with the
        values to be assigned in the label matrix. Zero means not a marker. If
        ``None`` (no markers given), the local minima of the image are used as
        markers.
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
           https://www.tu-chemnitz.de/etit/proaut/forschung/rsrc/cws_pSLIC_ICPR.pdf

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

    >>> labels = watershed(-distance, markers, mask=image)

    The algorithm works also for 3-D images, and can be used for example to
    separate overlapping spheres.
    """
    from ..segmentation import watershed as _watershed
    warn("skimage.morphology.watershed is deprecated and will be removed "
         "in version 0.19. Please use skimage.segmentation.watershed "
         "instead", FutureWarning, stacklevel=2)
    return _watershed(image, markers, connectivity, offset, mask,
                      compactness, watershed_line)


def label(input, neighbors=None, background=None, return_num=False,
          connectivity=None):
    r"""Label connected regions of an integer array.

    Two pixels are connected when they are neighbors and have the same value.
    In 2D, they can be neighbors either in a 1- or 2-connected sense.
    The value refers to the maximum number of orthogonal hops to consider a
    pixel/voxel a neighbor::

      1-connectivity     2-connectivity     diagonal connection close-up

           [ ]           [ ]  [ ]  [ ]             [ ]
            |               \  |  /                 |  <- hop 2
      [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
            |               /  |  \             hop 1
           [ ]           [ ]  [ ]  [ ]

    Parameters
    ----------
    input : ndarray of dtype int
        Image to label.
    neighbors : {4, 8}, int, optional
        Whether to use 4- or 8-"connectivity".
        In 3D, 4-"connectivity" means connected pixels have to share face,
        whereas with 8-"connectivity", they have to share only edge or vertex.
        **Deprecated, use** ``connectivity`` **instead.**
    background : int, optional
        Consider all pixels with this value as background pixels, and label
        them as 0. By default, 0-valued pixels are considered as background
        pixels.
    return_num : bool, optional
        Whether to return the number of assigned labels.
    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel
        as a neighbor.
        Accepted values are ranging from  1 to input.ndim. If ``None``, a full
        connectivity of ``input.ndim`` is used.

    Returns
    -------
    labels : ndarray of dtype int
        Labeled array, where all connected regions are assigned the
        same integer value.
    num : int, optional
        Number of labels, which equals the maximum label index and is only
        returned if return_num is `True`.

    See Also
    --------
    regionprops

    References
    ----------
    .. [1] Christophe Fiorio and Jens Gustedt, "Two linear time Union-Find
           strategies for image processing", Theoretical Computer Science
           154 (1996), pp. 165-181.
    .. [2] Kensheng Wu, Ekow Otoo and Arie Shoshani, "Optimizing connected
           component labeling algorithms", Paper LBNL-56864, 2005,
           Lawrence Berkeley National Laboratory (University of California),
           http://repositories.cdlib.org/lbnl/LBNL-56864

    Examples
    --------
    >>> import numpy as np
    >>> x = np.eye(3).astype(int)
    >>> print(x)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, connectivity=1))
    [[1 0 0]
     [0 2 0]
     [0 0 3]]
    >>> print(label(x, connectivity=2))
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, background=-1))
    [[1 2 2]
     [2 1 2]
     [2 2 1]]
    >>> x = np.array([[1, 0, 0],
    ...               [1, 1, 5],
    ...               [0, 0, 0]])
    >>> print(label(x))
    [[1 0 0]
     [1 1 2]
     [0 0 0]]
    """
    from ..measure import label as _label
    warn("skimage.morphology.label is deprecated and will be removed "
         "in version 0.19. Please use skimage.measure.label instead",
         FutureWarning, stacklevel=2)
    return _label(input, neighbors, background, return_num, connectivity)


def flood_fill(image, seed_point, new_value, *, selem=None, connectivity=None,
               tolerance=None, in_place=False, inplace=None):
    """Perform flood filling on an image.

    Starting at a specific `seed_point`, connected points equal or within
    `tolerance` of the seed value are found, then set to `new_value`.

    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    seed_point : tuple or int
        The point in `image` used as the starting point for the flood fill.  If
        the image is 1D, this point may be given as an integer.
    new_value : `image` type
        New value to set the entire fill.  This must be chosen in agreement
        with the dtype of `image`.
    selem : ndarray, optional
        A structuring element used to determine the neighborhood of each
        evaluated pixel. It must contain only 1's and 0's, have the same number
        of dimensions as `image`. If not given, all adjacent pixels are
        considered as part of the neighborhood (fully connected).
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if `selem` is
        not None.
    tolerance : float or int, optional
        If None (default), adjacent values must be strictly equal to the
        value of `image` at `seed_point` to be filled.  This is fastest.
        If a tolerance is provided, adjacent points with values within plus or
        minus tolerance from the seed point are filled (inclusive).
    in_place : bool, optional
        If True, flood filling is applied to `image` in place.  If False, the
        flood filled result is returned without modifying the input `image`
        (default).
    inplace : bool, optional
        This parameter is deprecated and will be removed in version 0.19.0
        in favor of in_place. If True, flood filling is applied to `image`
        inplace. If False, the flood filled result is returned without
        modifying the input `image` (default).

    Returns
    -------
    filled : ndarray
        An array with the same shape as `image` is returned, with values in
        areas connected to and equal (or within tolerance of) the seed point
        replaced with `new_value`.

    Notes
    -----
    The conceptual analogy of this operation is the 'paint bucket' tool in many
    raster graphics programs.

    Examples
    --------
    >>> from skimage.morphology import flood_fill
    >>> image = np.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = 1
    >>> image[3, 0] = 1
    >>> image[1:3, 4:6] = 2
    >>> image[3, 6] = 3
    >>> image
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, with full connectivity (diagonals included):

    >>> flood_fill(image, (1, 1), 5)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [5, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, excluding diagonal points (connectivity 1):

    >>> flood_fill(image, (1, 1), 5, connectivity=1)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill with a tolerance:

    >>> flood_fill(image, (0, 0), 5, tolerance=1)
    array([[5, 5, 5, 5, 5, 5, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 5, 5, 3]])
    """
    from ..segmentation import flood_fill
    warn("skimage.morphology.flood_fill is deprecated and will be removed "
         "in version 0.19. Please use skimage.segmentation.flood_fill instead",
         FutureWarning, stacklevel=2)
    return flood_fill(image, seed_point, new_value, selem,
                      connectivity, tolerance, in_place, inplace)


def flood(image, seed_point, *, selem=None, connectivity=None, tolerance=None):
    """Mask corresponding to a flood fill.

    Starting at a specific `seed_point`, connected points equal or within
    `tolerance` of the seed value are found.

    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    seed_point : tuple or int
        The point in `image` used as the starting point for the flood fill.  If
        the image is 1D, this point may be given as an integer.
    selem : ndarray, optional
        A structuring element used to determine the neighborhood of each
        evaluated pixel. It must contain only 1's and 0's, have the same number
        of dimensions as `image`. If not given, all adjacent pixels are
        considered as part of the neighborhood (fully connected).
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is larger or
        equal to `connectivity` are considered neighbors. Ignored if
        `selem` is not None.
    tolerance : float or int, optional
        If None (default), adjacent values must be strictly equal to the
        initial value of `image` at `seed_point`.  This is fastest.  If a value
        is given, a comparison will be done at every point and if within
        tolerance of the initial value will also be filled (inclusive).

    Returns
    -------
    mask : ndarray
        A Boolean array with the same shape as `image` is returned, with True
        values for areas connected to and equal (or within tolerance of) the
        seed point.  All other values are False.

    Notes
    -----
    The conceptual analogy of this operation is the 'paint bucket' tool in many
    raster graphics programs.  This function returns just the mask
    representing the fill.

    If indices are desired rather than masks for memory reasons, the user can
    simply run `numpy.nonzero` on the result, save the indices, and discard
    this mask.

    Examples
    --------
    >>> from skimage.morphology import flood
    >>> image = np.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = 1
    >>> image[3, 0] = 1
    >>> image[1:3, 4:6] = 2
    >>> image[3, 6] = 3
    >>> image
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, with full connectivity (diagonals included):

    >>> mask = flood(image, (1, 1))
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [5, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, excluding diagonal points (connectivity 1):

    >>> mask = flood(image, (1, 1), connectivity=1)
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill with a tolerance:

    >>> mask = flood(image, (0, 0), tolerance=1)
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[5, 5, 5, 5, 5, 5, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 5, 5, 3]])
    """
    from ..segmentation import flood
    warn("skimage.morphology.flood is deprecated and will be removed "
         "in version 0.19. Please use skimage.segmentation.flood instead",
         FutureWarning, stacklevel=2)
    return flood(image, seed_point, selem, connectivity, tolerance)
