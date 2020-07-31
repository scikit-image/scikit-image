import numpy as np
from itertools import product
from scipy.ndimage import generic_filter

from skimage.util import invert, view_as_windows
from ._rolling_ball_cy import apply_kernel


def rolling_ellipsoid(image, kernel_size=(100, 100), intensity_vertex=(100,)):
    """Estimate background intensity using a rolling ellipsoid.

    The rolling ellipsoid algorithm estimates background intensity for a
    grayscale image in case of uneven exposure. It is a generalization of the
    frequently used rolling ball algorithm [1]_.

    Parameters
    ----------
    image : array_like of rank 2, numeric
        The gray image to be filtered.
    kernel_size: array_like of rank 2, numeric
        The length of the special vertices of the ellipsoid.
    vertex : scalar, numeric
        The length of the intensity vertex of the ellipsoid.

    Notes
    -----
    For the pixel that has its background intensity estimated (wlog. at
    ``(0,0)``) the rolling ellipsoid method places an ellipsoid under it and
    raises it until it touches the image umbra at ``pos=(y,x)`` the background
    intensity is then estimated using the image intensity at that position
    (``image[*pos]``) plus the difference of ``intensity_vertex`` and the
    intensity of the ellipsoid at ``pos``. The intensity of the ellipsoid
    is computed using the canonical ellipsis equation:
        ``semi_spacial = kernel_size / 2``
        ``semi_vertex = intensity_vertex / 2``
        ``np.sum((pos/semi_spacial)**2) + (intensity/semi_vertex)**2 = 1``

    This algorithm assums that low intensity values (black) corresponds to the
    background. If you have a light background, invert the image before passing
    it into the function, e.g., using `utils.invert`.

    Returns
    -------
    filtered_image : ndarray of rank 3
        The image with background removed.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import data
    >>> from skimage.morphology import rolling_ball
    >>> image = data.coins()
    >>> result = rolling_ball(image, radius=100)

    References
    ----------
    .. [1] Sternberg, Stanley R. "Biomedical image processing." Computer 1
           (1983): 22-34. :DOI:`10.1109/MC.1983.1654163`
    """

    kernel_size = np.array(kernel_size)
    if not np.issubdtype(kernel_size.dtype, np.number):
        raise ValueError(
            f"kernel_size must be convertibale to a numeric array.")
    if not kernel_size.size == 2 and len(kernel_size.shape) == 2:
        raise ValueError(
            f"kernel_size must be a one dimensional array_like of size 2.")
    if np.any(kernel_size <= 0):
        raise ValueError(f"All elements of kernel_size must be greater zero.")
    kernel_size = kernel_size / 2

    intensity_vertex = np.array(intensity_vertex)
    if not np.issubdtype(intensity_vertex.dtype, np.number):
        raise ValueError(
            f"Intensity_vertex must be convertibale to a numeric array.")
    if not intensity_vertex.size == 1 and len(intensity_vertex.shape) == 1:
        raise ValueError(
            f"Intensity_vertex must be a scalar.")
    if np.any(intensity_vertex <= 0):
        raise ValueError(f"Intensity_vertex must be greater zero.")
    intensity_vertex = intensity_vertex / 2


    image = np.array(image)
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("Image must be of numeric type.")
    if not len(image.shape) == 2:
        raise ValueError("Image must be a three dimensional array.")
    img = image.copy().astype(float)

    # precompute ellipsoid intensity
    kernel_size_y, kernel_size_x = kernel_size
    tmp_x = np.arange(-kernel_size_x, kernel_size_x + 1)
    tmp_y = np.arange(-kernel_size_y, kernel_size_y + 1)
    x, y = np.meshgrid(tmp_x, tmp_y)
    tmp = (x / kernel_size_x) ** 2 + (y / kernel_size_y) ** 2
    cap_height = intensity_vertex - intensity_vertex * np.sqrt(
        np.clip(1 - tmp, 0, None)
    )

    kernel = np.array(tmp <= 1, dtype=float)
    kernel[kernel == 0] = np.Inf

    pad_amount = ((kernel.shape[0] // 2, kernel.shape[0] // 2),
                  (kernel.shape[1] // 2, kernel.shape[1] // 2))
    img = np.pad(img, pad_amount,  constant_values=np.Inf)

    windowed = view_as_windows(img, kernel.shape)

    # the implementation is very naive, but still surprisingly fast
    background = apply_kernel(windowed, kernel, cap_height)
    background = background.astype(image.dtype)

    filtered_image = image - background

    return filtered_image


def rolling_ball(image, radius=50):
    """Perform background subtraction using the rolling ball method.

    The rolling ball algorithm estimates background intensity of a grayscale
    image in case of uneven exposure. It is frequently used in biomedical
    image processing and was first proposed by Stanley R. Sternberg (1983) in
    the paper Biomedical Image Processing [1]_.

    Parameters
    ----------
    image : array_like of rank 2, numeric
        The gray image to be filtered.
    radius: scalar, numeric
        The radius of the ball/sphere rolled in the image.

    Notes
    -----
    If you are using images with a dtype other than `image.dtype == np.uint8`
    you may want to consider using `skimage.morphology.rolling_ellipsoid`
    instead. It allows you to specify different parameters for the spacial
    and intensity dimensions.

    This algorithm assums that low intensity values (black) corresponds to the
    background. If you have a light background, invert the image before passing
    it into the function, e.g., using `utils.invert`.

    Returns
    -------
    filtered_image : ndarray
        The image with background removed.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import data
    >>> from skimage.morphology import rolling_ball
    >>> image = data.coins()
    >>> result = rolling_ball(image, radius=100)

    References
    ----------
    .. [1] Sternberg, Stanley R. "Biomedical image processing." Computer 1
           (1983): 22-34. :DOI:`10.1109/MC.1983.1654163`
    """

    if not np.issubdtype(np.array(radius).dtype, np.number):
        raise ValueError("Radius must be a numeric type.")
    if radius <= 0:
        raise ValueError("Radius must be greater zero.")

    kernel = (radius * 2, radius * 2)
    intensity_vertex = radius * 2
    return rolling_ellipsoid(image, kernel, intensity_vertex)
