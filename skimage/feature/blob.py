import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace
import math
from math import sqrt, log
from scipy import spatial
from ..util import img_as_float
from .peak import peak_local_max
from ._hessian_det_appx import _hessian_matrix_det
from ..transform import integral_image
from .._shared.utils import check_nD


# This basic blob detection algorithm is based on:
# http://www.cs.utah.edu/~jfishbau/advimproc/project1/ (04.04.2013)
# Theory behind: https://en.wikipedia.org/wiki/Blob_detection (04.04.2013)


def _compute_disk_overlap(d, r1, r2):
    """
    Compute fraction of surface overlap between two disks of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.

    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first disk.
    r2 : float
        Radius of the second disk.

    Returns
    -------
    fraction: float
        Fraction of area of the overlap between the two disks.
    """

    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = np.clip(ratio1, -1, 1)
    acos1 = math.acos(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = np.clip(ratio2, -1, 1)
    acos2 = math.acos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 -
            0.5 * sqrt(abs(a * b * c * d)))
    return area / (math.pi * (min(r1, r2) ** 2))


def _compute_sphere_overlap(d, r1, r2):
    """
    Compute volume overlap fraction between two spheres of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.

    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first sphere.
    r2 : float
        Radius of the second sphere.

    Returns
    -------
    fraction: float
        Fraction of volume of the overlap between the two spheres.

    Notes
    -----
    See for example http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    for more details.
    """
    vol = (math.pi / (12 * d) * (r1 + r2 - d)**2 *
           (d**2 + 2 * d * (r1 + r2) - 3 * (r1**2 + r2**2) + 6 * r1 * r2))
    return vol / (4./3 * math.pi * min(r1, r2) ** 3)


def _blob_overlap(blob1, blob2, *, sigma_dim=1):
    """Finds the overlapping area fraction between two blobs.

    Returns a float representing fraction of overlapped area. Note that 0.0
    is *always* returned for dimension greater than 3.

    Parameters
    ----------
    blob1 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    blob2 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    sigma_dim : int, optional
        The dimensionality of the sigma value. Can be 1 or the same as the
        dimensionality of the blob space (2 or 3).

    Returns
    -------
    f : float
        Fraction of overlapped area (or volume in 3D).
    """
    ndim = len(blob1) - sigma_dim
    if ndim > 3:
        return 0.0
    root_ndim = sqrt(ndim)

    # we divide coordinates by sigma * sqrt(ndim) to rescale space to isotropy,
    # giving spheres of radius = 1 or < 1.
    if blob1[-1] == blob2[-1] == 0:
        return 0.0
    elif blob1[-1] > blob2[-1]:
        max_sigma = blob1[-sigma_dim:]
        r1 = 1
        r2 = blob2[-1] / blob1[-1]
    else:
        max_sigma = blob2[-sigma_dim:]
        r2 = 1
        r1 = blob1[-1] / blob2[-1]
    pos1 = blob1[:ndim] / (max_sigma * root_ndim)
    pos2 = blob2[:ndim] / (max_sigma * root_ndim)

    d = np.sqrt(np.sum((pos2 - pos1)**2))
    if d > r1 + r2:  # centers farther than sum of radii, so no overlap
        return 0.0

    # one blob is inside the other
    if d <= abs(r1 - r2):
        return 1.0

    if ndim == 2:
        return _compute_disk_overlap(d, r1, r2)

    else:  # ndim=3 http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        return _compute_sphere_overlap(d, r1, r2)


def _prune_blobs(blobs_array, overlap, *, sigma_dim=1):
    """Eliminated blobs with area overlap.

    Parameters
    ----------
    blobs_array : ndarray
        A 2d array with each row representing 3 (or 4) values,
        ``(row, col, sigma)`` or ``(pln, row, col, sigma)`` in 3D,
        where ``(row, col)`` (``(pln, row, col)``) are coordinates of the blob
        and ``sigma`` is the standard deviation of the Gaussian kernel which
        detected the blob.
        This array must not have a dimension of size 0.
    overlap : float
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.
    sigma_dim : int, optional
        The number of columns in ``blobs_array`` corresponding to sigmas rather
        than positions.

    Returns
    -------
    A : ndarray
        `array` with overlapping blobs removed.
    """
    sigma = blobs_array[:, -sigma_dim:].max()
    distance = 2 * sigma * sqrt(blobs_array.shape[1] - sigma_dim)
    tree = spatial.cKDTree(blobs_array[:, :-sigma_dim])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if _blob_overlap(blob1, blob2, sigma_dim=sigma_dim) > overlap:
                # note: this test works even in the anisotropic case because
                # all sigmas increase together.
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0

    return np.stack([b for b in blobs_array if b[-1] > 0])


def _format_exclude_border(img_ndim, exclude_border):
    """Format an ``exclude_border`` argument as a tuple of ints for calling
    ``peak_local_max``.
    """
    if isinstance(exclude_border, tuple):
        if len(exclude_border) != img_ndim:
            raise ValueError(
                "`exclude_border` should have the same length as the "
                "dimensionality of the image.")
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError(
                    "exclude border, when expressed as a tuple, must only "
                    "contain ints.")
        return exclude_border
    elif isinstance(exclude_border, int):
        return (exclude_border,) * img_ndim + (0,)
    elif exclude_border is True:
        raise ValueError("exclude_border cannot be True")
    elif exclude_border is False:
        return (0,) * (img_ndim + 1)
    else:
        raise ValueError(
            f"Unsupported value ({exclude_border}) for exclude_border"
        )


def blob_dog(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=2.0,
             overlap=.5, *, exclude_border=False):
    r"""Finds blobs in the given grayscale image.

    Blobs are found using the Difference of Gaussian (DoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Parameters
    ----------
    image : 2D or 3D ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : scalar or sequence of scalars, optional
        The minimum standard deviation for Gaussian kernel. Keep this low to
        detect smaller blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    max_sigma : scalar or sequence of scalars, optional
        The maximum standard deviation for Gaussian kernel. Keep this high to
        detect larger blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    sigma_ratio : float, optional
        The ratio between the standard deviation of Gaussian Kernels used for
        computing the Difference of Gaussians
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    exclude_border : tuple of ints, int, or False, optional
        If tuple of ints, the length of the tuple must match the input array's
        dimensionality.  Each element of the tuple will exclude peaks from
        within `exclude_border`-pixels of the border of the image along that
        dimension.
        If nonzero int, `exclude_border` excludes peaks from within
        `exclude_border`-pixels of the border of the image.
        If zero or False, peaks are identified regardless of their
        distance from the border.

    Returns
    -------
    A : (n, image.ndim + sigma) ndarray
        A 2d array with each row representing 2 coordinate values for a 2D
        image, and 3 coordinate values for a 3D image, plus the sigma(s) used.
        When a single sigma is passed, outputs are:
        ``(r, c, sigma)`` or ``(p, r, c, sigma)`` where ``(r, c)`` or
        ``(p, r, c)`` are coordinates of the blob and ``sigma`` is the standard
        deviation of the Gaussian kernel which detected the blob. When an
        anisotropic gaussian is used (sigmas per dimension), the detected sigma
        is returned for each dimension.

    See also
    --------
    skimage.filters.difference_of_gaussians

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_difference_of_Gaussians_approach

    Examples
    --------
    >>> from skimage import data, feature
    >>> feature.blob_dog(data.coins(), threshold=.5, max_sigma=40)
    array([[120.      , 272.      ,  16.777216],
           [193.      , 213.      ,  16.777216],
           [263.      , 245.      ,  16.777216],
           [185.      , 347.      ,  16.777216],
           [128.      , 154.      ,  10.48576 ],
           [198.      , 155.      ,  10.48576 ],
           [124.      , 337.      ,  10.48576 ],
           [ 45.      , 336.      ,  16.777216],
           [195.      , 102.      ,  16.777216],
           [125.      ,  45.      ,  16.777216],
           [261.      , 173.      ,  16.777216],
           [194.      , 277.      ,  16.777216],
           [127.      , 102.      ,  10.48576 ],
           [125.      , 208.      ,  10.48576 ],
           [267.      , 115.      ,  10.48576 ],
           [263.      , 302.      ,  16.777216],
           [196.      ,  43.      ,  10.48576 ],
           [260.      ,  46.      ,  16.777216],
           [267.      , 359.      ,  16.777216],
           [ 54.      , 276.      ,  10.48576 ],
           [ 58.      , 100.      ,  10.48576 ],
           [ 52.      , 155.      ,  16.777216],
           [ 52.      , 216.      ,  16.777216],
           [ 54.      ,  42.      ,  16.777216]])

    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}\sigma` for
    a 2-D image and :math:`\sqrt{3}\sigma` for a 3-D image.
    """
    image = img_as_float(image)

    # if both min and max sigma are scalar, function returns only one sigma
    scalar_sigma = np.isscalar(max_sigma) and np.isscalar(min_sigma)

    # Gaussian filter requires that sequence-type sigmas have same
    # dimensionality as image. This broadcasts scalar kernels
    if np.isscalar(max_sigma):
        max_sigma = np.full(image.ndim, max_sigma, dtype=float)
    if np.isscalar(min_sigma):
        min_sigma = np.full(image.ndim, min_sigma, dtype=float)

    # Convert sequence types to array
    min_sigma = np.asarray(min_sigma, dtype=float)
    max_sigma = np.asarray(max_sigma, dtype=float)

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                           for i in range(k + 1)])

    gaussian_images = [gaussian_filter(image, s) for s in sigma_list]

    # computing difference between two successive Gaussian blurred images
    # multiplying with average standard deviation provides scale invariance
    dog_images = [(gaussian_images[i] - gaussian_images[i + 1])
                  * np.mean(sigma_list[i]) for i in range(k)]

    image_cube = np.stack(dog_images, axis=-1)

    exclude_border = _format_exclude_border(image.ndim, exclude_border)
    local_maxima = peak_local_max(
        image_cube,
        threshold_abs=threshold,
        footprint=np.ones((3,) * (image.ndim + 1)),
        threshold_rel=0.0,
        exclude_border=exclude_border,
    )

    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))

    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]

    if scalar_sigma:
        # select one sigma column, keeping dimension
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

    # Remove sigma index and replace with sigmas
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])

    sigma_dim = sigmas_of_peaks.shape[1]

    return _prune_blobs(lm, overlap, sigma_dim=sigma_dim)


def blob_log(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=.2,
             overlap=.5, log_scale=False, *, exclude_border=False):
    r"""Finds blobs in the given grayscale image.

    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Parameters
    ----------
    image : 2D or 3D ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : scalar or sequence of scalars, optional
        the minimum standard deviation for Gaussian kernel. Keep this low to
        detect smaller blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    max_sigma : scalar or sequence of scalars, optional
        The maximum standard deviation for Gaussian kernel. Keep this high to
        detect larger blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.
    exclude_border : tuple of ints, int, or False, optional
        If tuple of ints, the length of the tuple must match the input array's
        dimensionality.  Each element of the tuple will exclude peaks from
        within `exclude_border`-pixels of the border of the image along that
        dimension.
        If nonzero int, `exclude_border` excludes peaks from within
        `exclude_border`-pixels of the border of the image.
        If zero or False, peaks are identified regardless of their
        distance from the border.

    Returns
    -------
    A : (n, image.ndim + sigma) ndarray
        A 2d array with each row representing 2 coordinate values for a 2D
        image, and 3 coordinate values for a 3D image, plus the sigma(s) used.
        When a single sigma is passed, outputs are:
        ``(r, c, sigma)`` or ``(p, r, c, sigma)`` where ``(r, c)`` or
        ``(p, r, c)`` are coordinates of the blob and ``sigma`` is the standard
        deviation of the Gaussian kernel which detected the blob. When an
        anisotropic gaussian is used (sigmas per dimension), the detected sigma
        is returned for each dimension.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian

    Examples
    --------
    >>> from skimage import data, feature, exposure
    >>> img = data.coins()
    >>> img = exposure.equalize_hist(img)  # improves detection
    >>> feature.blob_log(img, threshold = .3)
    array([[124.        , 336.        ,  11.88888889],
           [198.        , 155.        ,  11.88888889],
           [194.        , 213.        ,  17.33333333],
           [121.        , 272.        ,  17.33333333],
           [263.        , 244.        ,  17.33333333],
           [194.        , 276.        ,  17.33333333],
           [266.        , 115.        ,  11.88888889],
           [128.        , 154.        ,  11.88888889],
           [260.        , 174.        ,  17.33333333],
           [198.        , 103.        ,  11.88888889],
           [126.        , 208.        ,  11.88888889],
           [127.        , 102.        ,  11.88888889],
           [263.        , 302.        ,  17.33333333],
           [197.        ,  44.        ,  11.88888889],
           [185.        , 344.        ,  17.33333333],
           [126.        ,  46.        ,  11.88888889],
           [113.        , 323.        ,   1.        ]])

    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}\sigma` for
    a 2-D image and :math:`\sqrt{3}\sigma` for a 3-D image.
    """
    image = img_as_float(image)

    # if both min and max sigma are scalar, function returns only one sigma
    scalar_sigma = (
        True if np.isscalar(max_sigma) and np.isscalar(min_sigma) else False
    )

    # Gaussian filter requires that sequence-type sigmas have same
    # dimensionality as image. This broadcasts scalar kernels
    if np.isscalar(max_sigma):
        max_sigma = np.full(image.ndim, max_sigma, dtype=float)
    if np.isscalar(min_sigma):
        min_sigma = np.full(image.ndim, min_sigma, dtype=float)

    # Convert sequence types to array
    min_sigma = np.asarray(min_sigma, dtype=float)
    max_sigma = np.asarray(max_sigma, dtype=float)

    if log_scale:
        # for anisotropic data, we use the "highest resolution/variance" axis
        standard_axis = np.argmax(min_sigma)
        start = np.log10(min_sigma[standard_axis])
        stop = np.log10(max_sigma[standard_axis])
        scale = np.logspace(start, stop, num_sigma)[:, np.newaxis]
        sigma_list = scale * min_sigma / np.max(min_sigma)
    else:
        scale = np.linspace(0, 1, num_sigma)[:, np.newaxis]
        sigma_list = scale * (max_sigma - min_sigma) + min_sigma

    # computing gaussian laplace
    # average s**2 provides scale invariance
    gl_images = [-gaussian_laplace(image, s) * np.mean(s) ** 2
                 for s in sigma_list]

    image_cube = np.stack(gl_images, axis=-1)

    exclude_border = _format_exclude_border(image.ndim, exclude_border)
    local_maxima = peak_local_max(
        image_cube,
        threshold_abs=threshold,
        footprint=np.ones((3,) * (image.ndim + 1)),
        threshold_rel=0.0,
        exclude_border=exclude_border,
    )

    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))

    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]

    if scalar_sigma:
        # select one sigma column, keeping dimension
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

    # Remove sigma index and replace with sigmas
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])

    sigma_dim = sigmas_of_peaks.shape[1]

    return _prune_blobs(lm, overlap, sigma_dim=sigma_dim)


def blob_doh(image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.01,
             overlap=.5, log_scale=False):
    """Finds blobs in the given grayscale image.

    Blobs are found using the Determinant of Hessian method [1]_. For each blob
    found, the method returns its coordinates and the standard deviation
    of the Gaussian Kernel used for the Hessian matrix whose determinant
    detected the blob. Determinant of Hessians is approximated using [2]_.

    Parameters
    ----------
    image : 2D ndarray
        Input grayscale image.Blobs can either be light on dark or vice versa.
    min_sigma : float, optional
        The minimum standard deviation for Gaussian Kernel used to compute
        Hessian matrix. Keep this low to detect smaller blobs.
    max_sigma : float, optional
        The maximum standard deviation for Gaussian Kernel used to compute
        Hessian matrix. Keep this high to detect larger blobs.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect less prominent blobs.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.

    Returns
    -------
    A : (n, 3) ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel of the Hessian Matrix whose
        determinant detected the blob.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_determinant_of_the_Hessian

    .. [2] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,
           "SURF: Speeded Up Robust Features"
           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf

    Examples
    --------
    >>> from skimage import data, feature
    >>> img = data.coins()
    >>> feature.blob_doh(img)
    array([[197.        , 153.        ,  20.33333333],
           [124.        , 336.        ,  20.33333333],
           [126.        , 153.        ,  20.33333333],
           [195.        , 100.        ,  23.55555556],
           [192.        , 212.        ,  23.55555556],
           [121.        , 271.        ,  30.        ],
           [126.        , 101.        ,  20.33333333],
           [193.        , 275.        ,  23.55555556],
           [123.        , 205.        ,  20.33333333],
           [270.        , 363.        ,  30.        ],
           [265.        , 113.        ,  23.55555556],
           [262.        , 243.        ,  23.55555556],
           [185.        , 348.        ,  30.        ],
           [156.        , 302.        ,  30.        ],
           [123.        ,  44.        ,  23.55555556],
           [260.        , 173.        ,  30.        ],
           [197.        ,  44.        ,  20.33333333]])

    Notes
    -----
    The radius of each blob is approximately `sigma`.
    Computation of Determinant of Hessians is independent of the standard
    deviation. Therefore detecting larger blobs won't take more time. In
    methods line :py:meth:`blob_dog` and :py:meth:`blob_log` the computation
    of Gaussians for larger `sigma` takes more time. The downside is that
    this method can't be used for detecting blobs of radius less than `3px`
    due to the box filters used in the approximation of Hessian Determinant.
    """
    check_nD(image, 2)

    image = img_as_float(image)
    image = integral_image(image)

    if log_scale:
        start, stop = log(min_sigma, 10), log(max_sigma, 10)
        sigma_list = np.logspace(start, stop, num_sigma)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)

    hessian_images = [_hessian_matrix_det(image, s) for s in sigma_list]
    image_cube = np.dstack(hessian_images)

    local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
                                  footprint=np.ones((3,) * image_cube.ndim),
                                  threshold_rel=0.0,
                                  exclude_border=False)

    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))
    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)
    # Convert the last index to its corresponding scale value
    lm[:, -1] = sigma_list[local_maxima[:, -1]]
    return _prune_blobs(lm, overlap)
