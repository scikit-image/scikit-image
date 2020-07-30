import numpy as np
from itertools import product
from scipy.ndimage import generic_filter

from skimage.util import invert, view_as_windows
from ._rolling_ball_cy import apply_kernel


def rolling_ball(image, radius=50, white_background=False):
    """Perform background subtraction using the rolling ball method.

    The rolling ball algorithm estimates background intensity for a grayscale
    image in case of uneven exposure. It is frequently used in biomedical
    image processing and was first proposed by Stanley R. Sternberg (1983) in
    the paper Biomedical Image Processing [1]_.

    Parameters
    ----------
    image : ndarray
        The image to be filtered.
    radius : float or tuple, optional
        The radius of the ball that is used as the structuring element. If
        ``radius`` is a tuple it must be of the form
        ``(space_radius, intensity_radius)``, where `space_radius` indicates
        the radius of the ball along the spacial axis of the image and
        ``intensity_radius`` indicates the radius along the intensity axis.
        This is equivalent to rolling a a spheroid in the image (as opposed to
        a ball).
    white_background : bool, optional
        If true, the algorithm separates dark features from a bright
        background.

    Returns
    -------
    filtered_image : ndarray
        The image with background removed.
    background : ndarray
        The background that was removed.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import data
    >>> from skimage.morphology import rolling_ball

    Subtract a black background from an example image

    >>> image = data.coins()
    >>> result, bg = rolling_ball(image, radius=100)

    Subtract a white background from an example image

    >>> image = data.page()
    >>> result, bg = rolling_ball(image, radius=100, white_background=True)

    References
    ----------
    .. [1] Sternberg, Stanley R. "Biomedical image processing." Computer 1
           (1983): 22-34. :DOI:`10.1109/MC.1983.1654163`
    """

    try:
        if len(radius) == 2:
            spacial_radius = radius[0]
            intensity_vertex = radius[1]
    except TypeError:
        spacial_radius = radius
        intensity_vertex = radius
    except IndexError:
        # essentially radius need to have __len__ and __getitem__
        # so that we can extract the values
        raise ValueError(f"Radius must be a scalar or tuple-like.")

    try:
        spacial_radius = float(spacial_radius)
    except ValueError:
        raise ValueError(
            f"Radius should be float or float tuple, "
            f"was {type(spacial_radius)}"
        )

    try:
        intensity_vertex = float(intensity_vertex)
    except ValueError:
        raise ValueError(
            f"Radius should be float or float tuple, "
            f"was {type(intensity_vertex)}")

    if spacial_radius <= 0:
        raise ValueError(
            f"Spacial radius must be greater zero, was {spacial_radius}")

    if intensity_vertex <= 0:
        raise ValueError(
            f"Intensity radius must be greater zero, was {intensity_vertex}")

    try:
        white_background = bool(white_background)
    except ValueError:
        raise ValueError(
            f"white_background must be convertible to a boolean, "
            f"was {type(white_background)}"
        )

    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"image must be a np.ndarray, "
            f"was {type(image)}"
        )

    if not issubclass(image.dtype.type, np.integer):
        raise ValueError("Currently only integer images are supported.")

    img = image.copy()
    if white_background:
        img = invert(img)

    # assuming selem touches the umbra at (x,y) pre-compute the
    # (relative) height of selem at the center
    # (selem is a ball/spheroid)
    spacial_upper_bound = int(np.ceil(spacial_radius))
    L = np.arange(-spacial_upper_bound, spacial_upper_bound + 1)
    X, Y = np.meshgrid(L, L)
    distance = np.sqrt(X ** 2 + Y ** 2)
    sagitta = spacial_radius - spacial_radius * np.sqrt(
        np.clip(1 - (distance / intensity_vertex) ** 2, 0, None)
    )

    kernel = np.array(distance <= spacial_radius, dtype=float)
    kernel[kernel == 0] = np.Inf

    img = np.pad(img, spacial_upper_bound,
                 constant_values=np.iinfo(img.dtype).max)

    windowed = view_as_windows(img.astype(float), kernel.shape)

    # the implementation is very naive, but still surprisingly fast
    background = apply_kernel(windowed, kernel, sagitta)

    background = np.round(background).astype(image.dtype)

    if white_background:
        filtered_image = invert(image) - background
        filtered_image = invert(filtered_image)
    else:
        filtered_image = image - background

    return filtered_image
