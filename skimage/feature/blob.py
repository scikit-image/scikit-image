import math
import warnings
import inspect
from functools import wraps

import numpy as np
import scipy.ndimage as ndi
from scipy import spatial

from .._shared.filters import gaussian
from .._shared.utils import (
    _supported_float_type,
    check_nD,
    DEPRECATED,
)
from ..util import PendingSkimage2Change
from ..transform import integral_image
from ._hessian_det_appx import _hessian_matrix_det
from .peak import peak_local_max

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

    ratio1 = (d**2 + r1**2 - r2**2) / (2 * d * r1)
    ratio1 = np.clip(ratio1, -1, 1)
    acos1 = math.acos(ratio1)

    ratio2 = (d**2 + r2**2 - r1**2) / (2 * d * r2)
    ratio2 = np.clip(ratio2, -1, 1)
    acos2 = math.acos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = r1**2 * acos1 + r2**2 * acos2 - 0.5 * math.sqrt(abs(a * b * c * d))
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
    vol = (
        math.pi
        / (12 * d)
        * (r1 + r2 - d) ** 2
        * (d**2 + 2 * d * (r1 + r2) - 3 * (r1**2 + r2**2) + 6 * r1 * r2)
    )
    return vol / (4.0 / 3 * math.pi * min(r1, r2) ** 3)


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
    root_ndim = math.sqrt(ndim)

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

    d = np.sqrt(np.sum((pos2 - pos1) ** 2))
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
    distance = 2 * sigma * math.sqrt(blobs_array.shape[1] - sigma_dim)
    tree = spatial.cKDTree(blobs_array[:, :-sigma_dim])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for i, j in pairs:
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
                "dimensionality of the image."
            )
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError(
                    "exclude border, when expressed as a tuple, must only contain ints."
                )
        return exclude_border + (0,)
    elif isinstance(exclude_border, int):
        return (exclude_border,) * img_ndim + (0,)
    elif exclude_border is True:
        raise ValueError("exclude_border cannot be True")
    elif exclude_border is False:
        return (0,) * (img_ndim + 1)
    else:
        raise ValueError(f'Unsupported value ({exclude_border}) for exclude_border')


_THRESHOLD_WARNING = """{summary}

Starting with version 0.26, the parameter `threshold` is deprecated in favor of
`threshold_abs` that preserves the value range of `image`. Starting with
version 2.0, this includes leaving `threshold` unspecified and relying on its
default value which will be changed.  In version 2.2 (or later), this parameter
will be removed completely. Set `threshold_abs` explicitly to `None` or a valid
value to silence this warning. When switching to `threshold_abs`, if `image` is
of integer dtype, adjust the old `threshold` with:

    import numpy as np
    threshold_abs = threshold * {scale_factor_eq}

Hint: {hint}

For more details, see the migration guide:
https://scikit-image.org/docs/dev/user_guide/skimage2_migration.html#threshold-blob-funcs
"""


_THRESHOLD_REL_WARNING = """Must set parameter `threshold_rel` explicitly.

Starting in version 2.0 and skimage2, the default of the parameter
`threshold_rel` will change to the former value of the deprecated parameter
`threshold`. If you want to preserve the old behavior, set `threshold_rel=None`
explicitly.

For more details, see the migration guide:
https://scikit-image.org/docs/dev/user_guide/skimage2_migration.html#threshold-blob-funcs
"""


def _deprecate_threshold_with_scaling(func):
    """Warn if deprecated `threshold` is used or `threshold_abs` isn't.

    If the deprecated `threshold` is used implicitly or explicitly, its value
    is adjusted if necessary and passed to the new `threshold_abs` instead.
    `threshold` is then assigned `DEPRECATED`.

    `blob_doh` is special cased, because the scaling factor needs to be
    squared.
    """
    scale_factor_eq = "np.iinfo(image.dtype).max"

    def scale_threshold(image, threshold):
        if threshold is not None and image.dtype.kind in "ui":
            threshold *= np.iinfo(image.dtype).max
        return threshold

    if func.__name__ == "blob_doh":
        scale_factor_eq = "np.iinfo(image.dtype).max ** 2"

        def scale_threshold(image, threshold):
            if threshold is not None and image.dtype.kind in "ui":
                threshold *= np.iinfo(image.dtype).max ** 2
            return threshold

    @wraps(func)
    def wrapper(image, *args, **kwargs):
        if len(args) >= 5 or "threshold" in kwargs:
            # Deprecated `threshold` is passed explicitly

            threshold = kwargs["threshold"] if "threshold" in kwargs else args[4]
            threshold_abs = scale_threshold(image, threshold)

            msg = _THRESHOLD_WARNING.format(
                summary="Parameter `threshold` is deprecated.",
                hint=f"For `image` with dtype '{image.dtype}', "
                f"`{threshold=}` is equivalent to\n`{threshold_abs=}`.",
                scale_factor_eq=scale_factor_eq,
            )
            warnings.warn(msg, category=FutureWarning, stacklevel=2)

            if "threshold_abs" in kwargs:
                msg = (
                    "got value for deprecated argument `threshold` and "
                    "its successor `threshold_abs`"
                )
                raise TypeError(msg)

            # Replace threshold with `None`
            if "threshold" in kwargs:
                kwargs["threshold"] = DEPRECATED
            else:
                args = args[:4] + (DEPRECATED, *args[5:])
            kwargs["threshold_abs"] = threshold_abs

        elif "threshold_abs" not in kwargs:
            # Required `threshold_abs` is not given explicitly, extract old
            # default from `threshold` and scale
            sig = inspect.signature(func)
            threshold = sig.parameters["threshold"].default
            assert isinstance(threshold, float)
            threshold_abs = scale_threshold(image, threshold)

            kwargs["threshold"] = DEPRECATED
            kwargs["threshold_abs"] = threshold_abs

            # Default of deprecated `threshold` was used implicitly,
            # will change in skimage2
            msg = _THRESHOLD_WARNING.format(
                summary="Must set new parameter `threshold_abs` explicitly.",
                hint=f"For `image` with dtype '{image.dtype}', "
                f"`{threshold=}` is equivalent to\n`{threshold_abs=}`.",
                scale_factor_eq=scale_factor_eq,
            )
            warnings.warn(msg, category=PendingSkimage2Change, stacklevel=2)

        else:
            kwargs["threshold"] = DEPRECATED

        if "threshold_rel" not in kwargs:
            # Default of deprecated `threshold_rel` is used implicitly which
            # will change in skimage2
            warnings.warn(
                _THRESHOLD_REL_WARNING, category=PendingSkimage2Change, stacklevel=2
            )

        return func(image, *args, **kwargs)

    return wrapper


@_deprecate_threshold_with_scaling
def blob_dog(
    image,
    min_sigma=1,
    max_sigma=50,
    sigma_ratio=1.6,
    threshold=0.5,
    overlap=0.5,
    *,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=False,
):
    r"""Finds blobs in the given grayscale image.

    Blobs are found using the Difference of Gaussian (DoG) method [1]_, [2]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Parameters
    ----------
    image : ndarray
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
    threshold : float or None, optional, DEPRECATED!

        .. attention::
            The current behavior of this parameter rescales the value range of
            `image` based on its dtype with :func:`~.img_as_float`. This
            affects the value of maxima relative to this absolute value.

        .. deprecated:: 0.26.0
            This parameter is deprecated in favor of the new `threshold_abs`
            parameter which preservers the value range of `image`. This
            includes leaving `threshold` unspecified and relying on its default
            value. In version 2.2 (or later), this parameter will be removed
            completely. When switching to `threshold_abs`, if `image` is of
            integer dtype, adjust the old `threshold` with:

            .. code:: python

                threshold_abs = threshold * np.iinfo(image.dtype).max

    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    threshold_abs : float or None, optional
        Minimum absolute intensity of peaks. If `threshold_rel` is also
        specified, whichever threshold is larger will be used.

        .. versionadded:: 0.26.0
            Replaces the `threshold` parameter with a range preserving option.

    threshold_rel : float or None, optional
        Minimum relative intensity of peaks, calculated as
        ``max(dog_space) * threshold_rel``, where ``dog_space`` refers to the
        stack of Difference-of-Gaussian (DoG) images computed internally. This
        should have a value between 0 and 1. If `threshold_abs` is also
        specified, whichever threshold is larger will be used.
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
        image, or 3 coordinate values for a 3D image, plus the sigma(s) used.
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
    .. [2] Lowe, D. G. "Distinctive Image Features from Scale-Invariant
        Keypoints." International Journal of Computer Vision 60, 91–110 (2004).
        https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
        :DOI:`10.1023/B:VISI.0000029664.99615.94`

    Examples
    --------
    >>> from skimage import data, feature
    >>> coins = data.coins()
    >>> feature.blob_dog(coins, threshold_abs=12.75, min_sigma=10, max_sigma=40)
    array([[128., 155.,  10.],
           [198., 155.,  10.],
           [124., 338.,  10.],
           [127., 102.,  10.],
           [193., 281.,  10.],
           [126., 208.,  10.],
           [267., 115.,  10.],
           [197., 102.,  10.],
           [198., 215.,  10.],
           [123., 279.,  10.],
           [126.,  46.,  10.],
           [259., 247.,  10.],
           [196.,  43.,  10.],
           [ 54., 276.,  10.],
           [267., 358.,  10.],
           [ 58., 100.,  10.],
           [259., 305.,  10.],
           [185., 347.,  16.],
           [261., 174.,  16.],
           [ 46., 336.,  16.],
           [ 54., 217.,  10.],
           [ 55., 157.,  10.],
           [ 57.,  41.,  10.],
           [260.,  47.,  16.]])

    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}\sigma` for
    a 2-D image and :math:`\sqrt{3}\sigma` for a 3-D image.
    """
    assert threshold is DEPRECATED

    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    # if both min and max sigma are scalar, function returns only one sigma
    scalar_sigma = np.isscalar(max_sigma) and np.isscalar(min_sigma)

    # Gaussian filter requires that sequence-type sigmas have same
    # dimensionality as image. This broadcasts scalar kernels
    if np.isscalar(max_sigma):
        max_sigma = np.full(image.ndim, max_sigma, dtype=float_dtype)
    if np.isscalar(min_sigma):
        min_sigma = np.full(image.ndim, min_sigma, dtype=float_dtype)

    # Convert sequence types to array
    min_sigma = np.asarray(min_sigma, dtype=float_dtype)
    max_sigma = np.asarray(max_sigma, dtype=float_dtype)

    if sigma_ratio <= 1.0:
        raise ValueError('sigma_ratio must be > 1.0')

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio**i) for i in range(k + 1)])

    # computing difference between two successive Gaussian blurred images
    # to obtain an approximation of the scale invariant Laplacian of the
    # Gaussian operator
    dog_image_cube = np.empty(image.shape + (k,), dtype=float_dtype)
    gaussian_previous = gaussian(image, sigma=sigma_list[0], mode='reflect')
    for i, s in enumerate(sigma_list[1:]):
        gaussian_current = gaussian(image, sigma=s, mode='reflect')
        dog_image_cube[..., i] = gaussian_previous - gaussian_current
        gaussian_previous = gaussian_current

    # normalization factor for consistency in DoG magnitude
    sf = 1 / (sigma_ratio - 1)
    dog_image_cube *= sf

    exclude_border = _format_exclude_border(image.ndim, exclude_border)
    local_maxima = peak_local_max(
        dog_image_cube,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
        footprint=np.ones((3,) * (image.ndim + 1)),
    )

    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, image.ndim + (1 if scalar_sigma else image.ndim)))

    # Convert local_maxima to float64
    lm = local_maxima.astype(float_dtype)

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


@_deprecate_threshold_with_scaling
def blob_log(
    image,
    min_sigma=1,
    max_sigma=50,
    num_sigma=10,
    threshold=0.2,
    overlap=0.5,
    log_scale=False,
    *,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=False,
):
    r"""Finds blobs in the given grayscale image.

    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Parameters
    ----------
    image : ndarray
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
    threshold : float or None, optional, DEPRECATED!

        .. attention::
            The current behavior of this parameter rescales the value range of
            `image` based on its dtype with :func:`~.img_as_float`. This
            affects the value of maxima relative to this absolute value.

        .. deprecated:: 0.26.0
            This parameter is deprecated in favor of the new `threshold_abs`
            parameter which preservers the value range of `image`. This
            includes leaving `threshold` unspecified and relying on its default
            value. In version 2.2 (or later), this parameter will be removed
            completely. When switching to `threshold_abs`, if `image` is of
            integer dtype, adjust the old `threshold` with:

            .. code:: python

                threshold_abs = threshold * np.iinfo(image.dtype).max

    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.
    threshold_abs : float or None, optional
        Minimum absolute intensity of peaks. If `threshold_rel` is also
        specified, whichever threshold is larger will be used.

        .. versionadded:: 0.26.0
            Replaces the `threshold` parameter with a range preserving option.

    threshold_rel : float or None, optional
        Minimum intensity of peaks, calculated as
        ``max(log_space) * threshold_rel``, where ``log_space`` refers to the
        stack of Laplacian-of-Gaussian (LoG) images computed internally. This
        should have a value between 0 and 1. If None, `threshold` is used
        instead.
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
        image, or 3 coordinate values for a 3D image, plus the sigma(s) used.
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
    >>> feature.blob_log(img, threshold_abs= .3)
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
    assert threshold is DEPRECATED

    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    # if both min and max sigma are scalar, function returns only one sigma
    scalar_sigma = True if np.isscalar(max_sigma) and np.isscalar(min_sigma) else False

    # Gaussian filter requires that sequence-type sigmas have same
    # dimensionality as image. This broadcasts scalar kernels
    if np.isscalar(max_sigma):
        max_sigma = np.full(image.ndim, max_sigma, dtype=float_dtype)
    if np.isscalar(min_sigma):
        min_sigma = np.full(image.ndim, min_sigma, dtype=float_dtype)

    # Convert sequence types to array
    min_sigma = np.asarray(min_sigma, dtype=float_dtype)
    max_sigma = np.asarray(max_sigma, dtype=float_dtype)

    if log_scale:
        start = np.log10(min_sigma)
        stop = np.log10(max_sigma)
        sigma_list = np.logspace(start, stop, num_sigma)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)

    # computing gaussian laplace
    image_cube = np.empty(image.shape + (len(sigma_list),), dtype=float_dtype)
    for i, s in enumerate(sigma_list):
        # average s**2 provides scale invariance
        image_cube[..., i] = -ndi.gaussian_laplace(image, s) * np.mean(s) ** 2

    exclude_border = _format_exclude_border(image.ndim, exclude_border)
    local_maxima = peak_local_max(
        image_cube,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
        footprint=np.ones((3,) * (image.ndim + 1)),
    )

    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, image.ndim + (1 if scalar_sigma else image.ndim)))

    # Convert local_maxima to float64
    lm = local_maxima.astype(float_dtype)

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


@_deprecate_threshold_with_scaling
def blob_doh(
    image,
    min_sigma=1,
    max_sigma=30,
    num_sigma=10,
    threshold=0.01,
    overlap=0.5,
    log_scale=False,
    *,
    threshold_abs=None,
    threshold_rel=None,
):
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
    threshold : float or None, optional, DEPRECATED!

        .. attention::
            The current behavior of this parameter rescales the value range of
            `image` based on its dtype with :func:`~.img_as_float`. This
            affects the value of maxima relative to this absolute value.

        .. deprecated:: 0.26.0
            This parameter is deprecated in favor of the new `threshold_abs`
            parameter which preservers the value range of `image`. This
            includes leaving `threshold` unspecified and relying on its default
            value. In version 2.2 (or later), this parameter will be removed
            completely. When switching to `threshold_abs`, if `image` is of
            integer dtype, adjust the old `threshold` with:

            .. code:: python

                threshold_abs = threshold * np.iinfo(image.dtype).max ** 2

    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.
    threshold_abs : float or None, optional
        Minimum absolute intensity of peaks in the internally computed stack of
        Determinant-of-Hessian (DoH) images. Note that the amplitude
        relationship between `image` and DoH is cubic – so you need to square
        `threshold_abs` relative to values in `image`. If `threshold_rel` is
        also specified, whichever threshold is larger will be used.

        .. versionadded:: 0.26.0
            Replaces the `threshold` parameter with a range preserving option.

    threshold_rel : float or None, optional
        Minimum intensity of peaks, calculated as
        ``max(doh_space) * threshold_rel``, where ``doh_space`` refers to the
        stack of Determinant-of-Hessian (DoH) images computed internally. This
        should have a value between 0 and 1. If None, `threshold` is used
        instead.

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
    assert threshold is DEPRECATED

    check_nD(image, 2)

    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    image = integral_image(image)

    if log_scale:
        start, stop = math.log(min_sigma, 10), math.log(max_sigma, 10)
        sigma_list = np.logspace(start, stop, num_sigma)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)

    image_cube = np.empty(shape=image.shape + (len(sigma_list),), dtype=float_dtype)
    for j, s in enumerate(sigma_list):
        image_cube[..., j] = _hessian_matrix_det(image, s)

    local_maxima = peak_local_max(
        image_cube,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=False,
        footprint=np.ones((3,) * image_cube.ndim),
    )

    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))
    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)
    # Convert the last index to its corresponding scale value
    lm[:, -1] = sigma_list[local_maxima[:, -1]]
    return _prune_blobs(lm, overlap)
