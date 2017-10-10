import math
import numpy as np
from scipy import ndimage as ndi
from ..transform import resize
from ..util import img_as_float


def _smooth(image, sigma, mode, cval):
    """Return image with each channel smoothed by the Gaussian filter."""

    smoothed = np.empty(image.shape, dtype=np.double)

    # apply Gaussian filter to all dimensions independently
    if image.ndim == 3:
        for dim in range(image.shape[2]):
            ndi.gaussian_filter(image[..., dim], sigma,
                                output=smoothed[..., dim],
                                mode=mode, cval=cval)
    else:
        ndi.gaussian_filter(image, sigma, output=smoothed,
                            mode=mode, cval=cval)

    return smoothed


def _check_factor(factor):
    if factor <= 1:
        raise ValueError('scale factor must be greater than 1')


def pyramid_reduce(image, downscale=2, sigma=None, order=1,
                   mode='reflect', cval=0):
    """Smooth and then downsample image.

    Parameters
    ----------
    image : array
        Input image.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    out : array
        Smoothed and downsampled float image.

    References
    ----------
    .. [1] http://web.mit.edu/persci/people/adelson/pub_pdfs/pyramid83.pdf

    """

    _check_factor(downscale)

    image = img_as_float(image)

    rows = image.shape[0]
    cols = image.shape[1]
    out_rows = math.ceil(rows / float(downscale))
    out_cols = math.ceil(cols / float(downscale))

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    smoothed = _smooth(image, sigma, mode, cval)
    out = resize(smoothed, (out_rows, out_cols), order=order,
                 mode=mode, cval=cval)

    return out


def pyramid_expand(image, upscale=2, sigma=None, order=1,
                   mode='reflect', cval=0):
    """Upsample and then smooth image.

    Parameters
    ----------
    image : array
        Input image.
    upscale : float, optional
        Upscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * upscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of upsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    out : array
        Upsampled and smoothed float image.

    References
    ----------
    .. [1] http://web.mit.edu/persci/people/adelson/pub_pdfs/pyramid83.pdf

    """

    _check_factor(upscale)

    image = img_as_float(image)

    rows = image.shape[0]
    cols = image.shape[1]
    out_rows = math.ceil(upscale * rows)
    out_cols = math.ceil(upscale * cols)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * upscale / 6.0

    resized = resize(image, (out_rows, out_cols), order=order,
                     mode=mode, cval=cval)
    out = _smooth(resized, sigma, mode, cval)

    return out


def pyramid_gaussian(image, max_layer=-1, downscale=2, sigma=None, order=1,
                     mode='reflect', cval=0):
    """Yield images of the Gaussian pyramid formed by the input image.

    Recursively applies the `pyramid_reduce` function to the image, and yields
    the downscaled images.

    Note that the first image of the pyramid will be the original, unscaled
    image. The total number of images is `max_layer + 1`. In case all layers
    are computed, the last image is either a one-pixel image or the image where
    the reduction does not change its shape.

    Parameters
    ----------
    image : array
        Input image.
    max_layer : int
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers as float images.

    References
    ----------
    .. [1] http://web.mit.edu/persci/people/adelson/pub_pdfs/pyramid83.pdf

    """

    _check_factor(downscale)

    # cast to float for consistent data type in pyramid
    image = img_as_float(image)

    layer = 0
    rows = image.shape[0]
    cols = image.shape[1]

    prev_layer_image = image
    yield image

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    while layer != max_layer:
        layer += 1

        layer_image = pyramid_reduce(prev_layer_image, downscale, sigma, order,
                                     mode, cval)

        prev_rows = rows
        prev_cols = cols
        prev_layer_image = layer_image
        rows = layer_image.shape[0]
        cols = layer_image.shape[1]

        # no change to previous pyramid layer
        if prev_rows == rows and prev_cols == cols:
            break

        yield layer_image


def pyramid_laplacian(image, max_layer=-1, downscale=2, sigma=None, order=1,
                      mode='reflect', cval=0):
    """Yield images of the laplacian pyramid formed by the input image.

    Each layer contains the difference between the downsampled and the
    downsampled, smoothed image::

        layer = resize(prev_layer) - smooth(resize(prev_layer))

    Note that the first image of the pyramid will be the difference between the
    original, unscaled image and its smoothed version. The total number of
    images is `max_layer + 1`. In case all layers are computed, the last image
    is either a one-pixel image or the image where the reduction does not
    change its shape.

    Parameters
    ----------
    image : array
        Input image.
    max_layer : int
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers as float images.

    References
    ----------
    .. [1] http://web.mit.edu/persci/people/adelson/pub_pdfs/pyramid83.pdf
    .. [2] http://sepwww.stanford.edu/data/media/public/sep/morgan/texturematch/paper_html/node3.html

    """

    _check_factor(downscale)

    # cast to float for consistent data type in pyramid
    image = img_as_float(image)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    layer = 0
    rows = image.shape[0]
    cols = image.shape[1]

    smoothed_image = _smooth(image, sigma, mode, cval)
    yield image - smoothed_image

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    while layer != max_layer:
        layer += 1

        out_rows = math.ceil(rows / float(downscale))
        out_cols = math.ceil(cols / float(downscale))

        resized_image = resize(smoothed_image, (out_rows, out_cols),
                               order=order, mode=mode, cval=cval)
        smoothed_image = _smooth(resized_image, sigma, mode, cval)

        prev_rows = rows
        prev_cols = cols
        rows = resized_image.shape[0]
        cols = resized_image.shape[1]

        # no change to previous pyramid layer
        if prev_rows == rows and prev_cols == cols:
            break

        yield resized_image - smoothed_image
