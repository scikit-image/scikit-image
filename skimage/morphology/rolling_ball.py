import numpy as np
from numpy.lib.stride_tricks import as_strided

from skimage.util import invert, view_as_windows
from ._rolling_ball_cy import apply_kernel, apply_kernel_nan


def rolling_ellipsoid(image, kernel_shape=100, intensity_vertex=100,
                      has_nan=False, num_threads=None):
    """Estimate background intensity using a rolling ellipsoid.

    The rolling ellipsoid algorithm estimates background intensity for a
    grayscale image in case of uneven exposure. It is a generalization of the
    frequently used rolling ball algorithm [1]_.

    Parameters
    ----------
    image : ndarray
        The image to be filtered.
    kernel_shape: ndarray or scalar, optional
        The length of the non-intensity vertices of the ellipsoid. If
        ``kernel_shape`` is a ndarray, it must have the same dimensions as
        ``image``. If ``kernel_shape`` is a scalar it will be extended to
        have the same dimension via
        ``kernel_shape = kernel_shape * np.ones_like(image.shape)``.
        All elements must be greater than 0.
    intensity_vertex : scalar, optional
        The length of the intensity vertex of the ellipsoid. Must be greater
        than 0.
    has_nan: bool, optional
        If ``False`` (default) assumes that none of the values in ``image``
        are ``np.nan``, and uses a faster implementation.
    num_threads: int, optional
        The maximum number of threads to use. If ``None`` use the OpenMP
        default value; typically equal to the maximum number of virtual cores.
        Note: This is an upper limit to the number of threads. The exact number
        is determined by the system's OpenMP library.

    Returns
    -------
    background : ndarray
        The estimated background of the image.

    Notes
    -----

    - For the pixel that has its background intensity estimated (without loss
      of generality at ``(0,0)``) the rolling ellipsoid method places an
      ellipsoid under it and raises the ellipsoid until its surface touches the
      image umbra at ``pos=(y,x)``. The background intensity is then estimated
      using the image intensity at that position (``image[y, x]``) plus the
      difference of ``intensity_vertex`` and the surface of the ellipsoid at
      ``pos``. The surface intensity of the ellipsoid is computed using the
      canonical ellipsis equation::

            semi_spatial = kernel_shape / 2
            semi_vertex = intensity_vertex / 2
            np.sum((pos/semi_spatial)**2) + (intensity/semi_vertex)**2 = 1

    - This algorithm assumes that dark pixels correspond to the background. If
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
    >>> image = data.coins()
    >>> background = rolling_ellipsoid(data.coins())
    >>> filtered_image = image - background
    """

    if num_threads is None:
        num_threads = 0

    kernel_shape = np.asarray(kernel_shape)
    if kernel_shape.ndim == 0:
        kernel_shape = kernel_shape * np.ones_like(image.shape)

    kernel_shape_int = np.asarray(kernel_shape//2*2+1, dtype=np.intp)

    intensity_vertex = np.asarray(intensity_vertex, dtype=np.float_)
    intensity_vertex = intensity_vertex / 2

    image = np.asarray(image)
    img = image.astype(np.float_)

    ellipsoid_coords = np.stack(
        np.meshgrid(
            *[range(-x, x+1) for x in kernel_shape_int//2],
            indexing='ij'
        ),
        axis=-1).reshape(-1, len(kernel_shape))
    tmp = np.sum(
        (ellipsoid_coords / (kernel_shape[np.newaxis, :]/2)) ** 2, axis=1)
    ellipsoid_intensity = intensity_vertex - \
        intensity_vertex * np.sqrt(np.clip(1 - tmp, 0, None))
    ellipsoid_intensity = ellipsoid_intensity.astype(np.float_)

    pad_amount = np.round(kernel_shape / 2).astype(int)
    img = np.pad(img, pad_amount[:, np.newaxis],
                 constant_values=np.Inf, mode="constant")

    kernel = np.where(tmp <= 1, 1, np.inf)

    if has_nan:
        background = apply_kernel_nan(
            img.ravel(),
            kernel,
            ellipsoid_intensity,
            np.array(image.shape, dtype=np.intp),
            np.array(img.shape, dtype=np.intp),
            kernel_shape_int,
            num_threads
        )
    else:
        background = apply_kernel(
            img.ravel(),
            kernel,
            ellipsoid_intensity,
            np.array(image.shape, dtype=np.intp),
            np.array(img.shape, dtype=np.intp),
            kernel_shape_int,
            num_threads
        )

    background = background.astype(image.dtype)

    return background


def rolling_ball(image, radius=50, **kwargs):
    """
    Estimate background intensity using a rolling ball.

    This is a convenience function for the frequently used special case of
    ``rolling_ellipsoid`` where the spatial vertices and intensity vertex
    have the same value resulting in a spherical kernel. For details see
    ``rolling_ellipsoid``.

    Parameters
    ----------
    image : ndarray
        The image to be filtered.
    radius: scalar, numeric, optional
        The radius of the ball/sphere rolled in the image. Must be greater
        than 0.

    Returns
    -------
    background : ndarray
        The estimated background of the image.

    See Also
    --------
    rolling_ellipsoid :
        additional keyword arguments
        generalization to elliptical kernels; used internally


    Notes
    -----
    - If you are using images with a dtype other than `image.dtype == np.uint8`
      you may want to consider using `skimage.morphology.rolling_ellipsoid`
      instead. It allows you to specify different parameters for the spatial
      and intensity dimensions.
    - The ball has the same dimensionality as the input image. If you
      want to apply the filter plane-wise to a 3D image use ::
        kernel_shape = (1, 2 * radius, 2 * radius)
        intensity_vertex = 2 * radius
        rolling_ellipsoid(image, kernel_shape, intensity_vertex)``
      instead.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import data
    >>> from skimage.morphology import rolling_ball
    >>> image = data.coins()
    >>> background = rolling_ball(image, radius=200)
    >>> filtered_image = image - background
    """

    kernel_shape = radius * 2
    intensity_vertex = radius * 2
    return rolling_ellipsoid(image, kernel_shape, intensity_vertex, **kwargs)
