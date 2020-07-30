import numpy as np

from skimage.util import invert


def rolling_ball(image, radius=50, white_background=False):
    """Perform background subtraction using the rolling ball method.

    The rolling ball filter is a segmentation method that aims to separate the
    background from a grayscale image in case of uneven exposure. It is
    frequently used in biomedical image processing and was first proposed by
    Stanley R. Sternberg (1983) in the paper Biomedical Image Processing [1]_.

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
            space_vertex = radius[0]
            intensity_vertex = radius[1]
    except TypeError:
        space_vertex = radius
        intensity_vertex = radius
    except IndexError:
        # essentially radius need to have __len__ and __getitem__
        # so that we can extract the values
        raise ValueError(f"Radius must be a scalar or tuple-like.")

    try:
        space_vertex = float(space_vertex)
    except ValueError:
        raise ValueError(
            f"Radius should be float or float tuple, was {type(space_vertex)}")

    try:
        intensity_vertex = float(intensity_vertex)
    except ValueError:
        raise ValueError(
            f"Radius should be float or float tuple, "
            f"was {type(intensity_vertex)}")

    if space_vertex <= 0:
        raise ValueError(
            f"Spacial radius must be greater zero, was {space_vertex}")

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
            f"input_img must be a np.ndarray, "
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
    L = np.arange(-space_vertex, space_vertex + 1)
    X, Y = np.meshgrid(L, L)
    distance = np.sqrt(X ** 2 + Y ** 2)
    sagitta = space_vertex - space_vertex * np.sqrt(
        np.clip(1 - (distance ** 2 / intensity_vertex ** 2), 0, None)
    )

    kernel = np.array(distance <= space_vertex, dtype=float)
    kernel[kernel == 0] = np.Inf

    kernel_size = np.array(kernel.shape)
    img_original = img.copy()
    img_size = img.shape
    x = img
    stride = (1, 1)

    # pad the image
    padding_amount = kernel_size // 2 * 2
    half_pad = padding_amount // 2
    img = np.Inf * np.ones(x.shape + padding_amount, dtype=x.dtype)
    img[half_pad[0]:-half_pad[0], half_pad[1]:-half_pad[1]] = x
    large_img_size = img.shape

    # the following three blocks implement a variant of 2d convolution
    # in python. Each time a kernel is applied to a window, sagitta is added
    # to the respective pixel values and the reduction is done using `min` instead
    # of `sum`

    # window of affected pixel indices relative to anchor
    x_idx = np.arange(kernel_size[1])
    y_idx = large_img_size[1] * np.arange(kernel_size[0])
    kernel_idx = (x_idx[np.newaxis, :] + y_idx[:, np.newaxis]).flatten()

    # indices of each ancor for a kernel
    # (top left corner instead of center)
    x_anchors = np.arange(
        large_img_size[1] - kernel_size[1] + 1, step=stride[1])
    y_anchors = large_img_size[1] * \
        np.arange(large_img_size[0] - kernel_size[0] + 1, step=stride[0])
    anchor_offsets = (x_anchors[np.newaxis, :] +
                      y_anchors[:, np.newaxis]).flatten()

    # compute px indices in the image and apply the function/kernel
    # large images or kernel sizes don't fit into memory
    # do it in batches instead
    background = np.zeros(img_original.size)
    batch_size = int(2 ** 10)
    flat_img = img.flatten()
    flat_sagitta = sagitta.flatten()
    flat_kernel = kernel.flatten()
    for low in range(0, len(anchor_offsets), batch_size):
        high = np.minimum(low + batch_size, len(anchor_offsets))
        filter_idx = anchor_offsets[low:high,
                                    np.newaxis] + kernel_idx[np.newaxis, :]
        background_partial = np.min(
            (flat_img[filter_idx] + flat_sagitta[np.newaxis, :]) * flat_kernel[np.newaxis, :], axis=1)
        background[low:high] = background_partial

    background = background.reshape(img_size).astype(img_original.dtype)
    filtered_image = img_original - background

    if white_background:
        filtered_image = invert(filtered_image)

    return filtered_image
