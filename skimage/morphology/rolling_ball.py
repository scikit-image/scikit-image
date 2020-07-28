#!/usr/bin/env python

import numpy as np

from skimage.util import invert


def rolling_ball(input_img, radius=50, white_background=False):
    """Perform background subtraction using the rolling ball method.

    The rolling ball filter is a segmentation method that aims to separate the
    background from a grayscale image in case of uneven exposure. It is
    frequently used in biomedical image processing and was first proposed by
    Stanley R. Sternberg (1983) in the paper Biomedical Image Processing [1]_.

    Parameters
    ----------
    image : ndarray
        The image to be filtered.
    radius : float
        The radius of the ball that is used as the structuring element.
    white_background : bool
        If true, the algorithm separates dark features from a bright
        background.

    Returns
    -------
    filtered_image : ndarray
        The image with background removes
    background : ndarray
        The background that was removed.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import reconstruction

    First, we create a sinusoidal mask image with peaks at middle and ends.

    >>> x = np.linspace(0, 4 * np.pi)
    >>> y_mask = np.cos(x)

    Then, we create a seed image initialized to the minimum mask value (for
    reconstruction by dilation, min-intensity values don't spread) and add
    "seeds" to the left and right peak, but at a fraction of peak value (1).

    >>> y_seed = y_mask.min() * np.ones_like(x)
    >>> y_seed[0] = 0.5
    >>> y_seed[-1] = 0
    >>> y_rec = reconstruction(y_seed, y_mask)

    The reconstructed image (or curve, in this case) is exactly the same as the
    mask image, except that the peaks are truncated to 0.5 and 0. The middle
    peak disappears completely: Since there were no seed values in this peak
    region, its reconstructed value is truncated to the surrounding value (-1).

    As a more practical example, we try to extract the bright features of an
    image by subtracting a background image created by reconstruction.

    >>> y, x = np.mgrid[:20:0.5, :20:0.5]
    >>> bumps = np.sin(x) + np.sin(y)

    To create the background image, set the mask image to the original image,
    and the seed image to the original image with an intensity offset, `h`.

    >>> h = 0.3
    >>> seed = bumps - h
    >>> background = reconstruction(seed, bumps)

    The resulting reconstructed image looks exactly like the original image,
    but with the peaks of the bumps cut off. Subtracting this reconstructed
    image from the original image leaves just the peaks of the bumps

    >>> hdome = bumps - background

    This operation is known as the h-dome of the image and leaves features
    of height `h` in the subtracted image.

    Notes
    -----
    Check out the gallery example of this function (TODO: link)

    References
    ----------
    .. [1] Sternberg, Stanley R. "Biomedical image processing." Computer 1
           (1983): 22-34. :DOI:`10.1109/MC.1983.1654163`
    """

    if not isinstance(radius, float):
        try:
            radius = float(radius)
        except ValueError:
            raise ValueError(f"Radius should be float, was {type(radius)}")

    if radius <= 0:
        raise ValueError(f"Radius must be greater zeros, was {radius}")

    try:
        white_background = bool(white_background)
    except ValueError:
        raise ValueError(
            f"white_background must be convertible to a boolean, "
            f"was {type(white_background)}"
        )

    img = input_img.copy()
    if white_background:
        img = invert(img)

    # sagitta assuming the position is where the ball touches
    # the image umbra
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    distance = np.sqrt(X ** 2 + Y ** 2)
    sagitta = radius - np.sqrt(radius ** 2 - distance ** 2)
    sagitta[np.isnan(sagitta)] = np.Inf

    kernel = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=float)
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

    # indices of kernel in image coords
    x_idx = np.arange(kernel_size[1])
    y_idx = large_img_size[1] * np.arange(kernel_size[0])
    kernel_idx = (x_idx[np.newaxis, :] + y_idx[:, np.newaxis]).flatten()

    # indices corresponding to each ancor of the kernel
    # (top left corner instead of center)
    x_ancors = np.arange(
        large_img_size[1] - kernel_size[1] + 1, step=stride[1])
    y_ancors = large_img_size[1] * \
        np.arange(large_img_size[0] - kernel_size[0] + 1, step=stride[0])
    ancor_offsets = (x_ancors[np.newaxis, :] +
                     y_ancors[:, np.newaxis]).flatten()

    # large images or kernel sizes don't fit into memory
    # do it in batches instead
    background = np.zeros(img_original.size)
    batch_size = int(2 ** 8)
    flat_img = img.flatten()
    flat_sagitta = sagitta.flatten()
    for low in range(0, len(ancor_offsets), batch_size):
        high = np.minimum(low + batch_size, len(ancor_offsets)-1)
        filter_idx = ancor_offsets[low:high,
                                   np.newaxis] + kernel_idx[np.newaxis, :]
        background_partial = np.min(
            flat_img[filter_idx] + flat_sagitta[np.newaxis, :], axis=1)
        background[low:high] = background_partial

    background = background.reshape(img_size)
    filtered_image = img_original - background

    if white_background:
        filtered_image = invert(filtered_image)
        background = invert(background)

    return filtered_image, background
