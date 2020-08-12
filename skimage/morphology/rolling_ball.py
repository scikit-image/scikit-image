import numpy as np
from numpy.lib.stride_tricks import as_strided

from skimage.util import invert, view_as_windows
from ._rolling_ball_cy import apply_kernel, apply_kernel_nan, apply_kernel_flat


def rolling_ellipsoid(image, kernel_size=(100, 100), intensity_vertex=100,
                      has_nan=False):
    """Estimate background intensity using a rolling ellipsoid.

    The rolling ellipsoid algorithm estimates background intensity for a
    grayscale image in case of uneven exposure. It is a generalization of the
    frequently used rolling ball algorithm [1]_.

    Parameters
    ----------
    image : ndarray
        The last two dimensions are treated as the columns and rows of the
        image. The operation is broadcasted along the remaining ones.
    kernel_size: two-element tuple, numeric, optional
        The length of the spatial vertices of the ellipsoid.
    intensity_vertex : scalar, numeric, optional
        The length of the intensity vertex of the ellipsoid.
    has_nan: bool, optional
        If ``False`` (default) assumes that none of the values in ``image``
        are ``np.nan``, and uses a faster implementation.

    Returns
    -------
    filtered_image : ndarray
        The image with background removed.

    Notes
    -----

    - For the pixel that has its background intensity estimated (at ``(0,0)``)
      the rolling ellipsoid method places an ellipsoid under it and
      raises the ellipsoid until it touches the image umbra at ``pos=(y,x)``.
      The background intensity is then estimated using the image intensity at
      that position (``image[y, x]``) plus the difference of
      ``intensity_vertex`` and the intensity of the ellipsoid at ``pos``. The
      intensity of the ellipsoid is computed using the canonical ellipsis
      equation::

            semi_spatial = kernel_size / 2
            semi_vertex = intensity_vertex / 2
            np.sum((pos/semi_spatial)**2) + (intensity/semi_vertex)**2 = 1

    - This algorithm assums that dark pixels correspond to the background. If
      you have a bright background, invert the image before passing it to the
      function, e.g., using `utils.invert`.
    - This algorithm is sensitive to noise (in particular salt-and-pepper
      noise). If this is a problem in your image, you can apply mild
      gaussian smoothing before passing the image to this function.

    References
    ----------
    .. [1] Sternberg, Stanley R. "Biomedical image processing." Computer 1
           (1983): 22-34. :DOI:`10.1109/MC.1983.1654163`

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import data
    >>> from skimage.morphology import rolling_ellipsoid
    >>> result = rolling_ellipsoid(data.coins())
    """

    kernel_size = np.asarray(kernel_size)
    if not np.issubdtype(kernel_size.dtype, np.number):
        raise ValueError(
            "kernel_size must be convertible to a numeric array.")
    if not kernel_size.shape == (2,):
        raise ValueError(
            "kernel_size must be a two element tuple.")
    if np.any(kernel_size <= 0):
        raise ValueError("All elements of kernel_size must be greater zero.")
    kernel_size = kernel_size / 2

    intensity_vertex = np.asarray(intensity_vertex, dtype=np.float_)
    if not intensity_vertex.shape == tuple():
        raise ValueError("Intensity_vertex must be a scalar.")
    if np.any(intensity_vertex <= 0):
        raise ValueError("Intensity_vertex must be greater zero.")
    intensity_vertex = intensity_vertex / 2

    image = np.asarray(image)
    img = image.astype(np.float_)

    kernel_size_y, kernel_size_x = np.round(kernel_size).astype(int)

    pad_amount = [(0, 0)] * image.ndim
    pad_amount[-2] = (kernel_size_y, kernel_size_y)
    pad_amount[-1] = (kernel_size_x, kernel_size_x)
    img = np.pad(img, pad_amount,  constant_values=np.Inf, mode="constant")

    tmp_x = np.arange(-kernel_size_x, kernel_size_x + 1)
    tmp_y = np.arange(-kernel_size_y, kernel_size_y + 1)
    x, y = np.meshgrid(tmp_x, tmp_y)

    kernel_size_y, kernel_size_x = kernel_size
    tmp = (x / kernel_size_x) ** 2 + (y / kernel_size_y) ** 2
    cap_height = intensity_vertex - intensity_vertex * \
        np.sqrt(np.clip(1 - tmp, 0, None))
    cap_height = cap_height.astype(np.float_)

    kernel = np.asarray(tmp <= 1, dtype=np.float_)
    kernel[kernel == 0] = np.Inf

    if has_nan:
        strides = (img.itemsize, img.strides[-2], img.itemsize)
        shape = (img.size - (kernel.shape[0] - 1) * img.shape[-1] -
                 (kernel.shape[1] - 1), *kernel.shape)
        windowed = as_strided(img, shape, strides)
        background = apply_kernel_nan(windowed, kernel, cap_height)
        background = as_strided(background, image.shape, img.strides)
    else:
        # windowed = view_as_windows(img, kernel.shape)
        # background = apply_kernel(windowed, kernel, cap_height)
        strides = (img.itemsize, img.strides[-2], img.itemsize)
        shape = (img.size - (kernel.shape[0] - 1) * img.shape[-1] -
                 (kernel.shape[1] - 1), *kernel.shape)
        windowed = as_strided(img, shape, strides)
        background = apply_kernel_flat(windowed, kernel, cap_height)
        background = as_strided(background, image.shape, img.strides)

    background = background.astype(image.dtype)

    filtered_image = image - background

    return filtered_image


def rolling_ball(image, radius=50, has_nan=False):
    """
    Estimate background intensity using a rolling ball.

    This is a convenience function for the frequently used special case of
    ``rolling_ellipsoid`` where the spacial vertices and intensity vertex
    have the same value resulting in a spherical kernel. For details see
    ``rolling_ellipsoid``.

    Parameters
    ----------
    image : (N, M) ndarray
        The gray image to be filtered.
    radius: scalar, numeric, optional
        The radius of the ball/sphere rolled in the image.
    has_nan: bool, optional
        If ``False`` (default) assumes that none of the values in ``image``
        are ``np.nan``, and uses a faster implementation.

    Returns
    -------
    filtered_image : ndarray
        The image with background removed.

    See Also
    --------
    rolling_ellipsoid : generalization to elliptical kernels; used internally

    Notes
    -----
    - If you are using images with a dtype other than `image.dtype == np.uint8`
      you may want to consider using `skimage.morphology.rolling_ellipsoid`
      instead. It allows you to specify different parameters for the spacial
      and intensity dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import data
    >>> from skimage.morphology import rolling_ball
    >>> result = rolling_ball(data.coins(), radius=200)
    """

    if not np.issubdtype(np.asarray(radius).dtype, np.number):
        raise ValueError("Radius must be a numeric type.")
    if radius <= 0:
        raise ValueError("Radius must be greater zero.")

    kernel = (radius * 2, radius * 2)
    intensity_vertex = radius * 2
    return rolling_ellipsoid(image, kernel, intensity_vertex, has_nan)
