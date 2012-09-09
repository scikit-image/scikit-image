import math
import numpy as np
from scipy import ndimage
from skimage.transform import resize
from skimage.util import img_as_float


def _smooth(image, sigma, mode, cval):
    """Return image with each channel smoothed by the gaussian filter."""

    smoothed = np.empty(image.shape, dtype=np.double)

    if image.ndim == 3: # apply gaussian filter to all dimensions independently
        for dim in range(image.shape[2]):
            ndimage.gaussian_filter(image[..., dim], sigma,
                                    output=smoothed[..., dim],
                                    mode=mode, cval=cval)
    else:
        ndimage.gaussian_filter(image, sigma, output=smoothed,
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
        Sigma for gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `scipy.ndimage.map_coordinates` for detail.
    mode :  {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    out : array
        Smoothed and downsampled image.

    References
    ----------
    ..[1] http://web.mit.edu/persci/people/adelson/pub_pdfs/pyramid83.pdf

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
        Sigma for gaussian filter. Default is `2 * upscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `scipy.ndimage.map_coordinates` for detail.
    mode :  {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    out : array
        Upsampled and smoothed image.

    References
    ----------
    ..[1] http://web.mit.edu/persci/people/adelson/pub_pdfs/pyramid83.pdf

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


def build_gaussian_pyramid(image, max_layer=-1, downscale=2, sigma=None,
                           order=1, mode='reflect', cval=0):
    """Build gaussian pyramid.

    Recursively applies the `pyramid_reduce` function to the image.

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
        Sigma for gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `scipy.ndimage.map_coordinates` for detail.
    mode :  {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers.

    References
    ----------
    ..[1] http://web.mit.edu/persci/people/adelson/pub_pdfs/pyramid83.pdf

    """

    _check_factor(downscale)

    image = img_as_float(image)

    layer = 0
    rows = image.shape[0]
    cols = image.shape[1]

    # cast to float for consistent data type in pyramid
    prev_layer_image = image
    yield image

    # build downsampled images until max_layer is reached or downsampled image
    # has size of 1 in one direction
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


def build_laplacian_pyramid(image, max_layer=-1, downscale=2, sigma=None,
                            order=1, mode='reflect', cval=0):
    """Build laplacian pyramid.

    Each layer contains the difference between the downsampled and the
    downsampled plus smoothed image.

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
        Sigma for gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `scipy.ndimage.map_coordinates` for detail.
    mode :  {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers.

    References
    ----------
    ..[1] http://web.mit.edu/persci/people/adelson/pub_pdfs/pyramid83.pdf

    """

    _check_factor(downscale)

    image = img_as_float(image)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    layer = 0
    rows = image.shape[0]
    cols = image.shape[1]

    prev_layer_image = image - _smooth(image, sigma, mode, cval)
    yield prev_layer_image

    # build downsampled images until max_layer is reached or downsampled image
    # has size of 1 in one direction
    while layer != max_layer:
        layer += 1

        rows = prev_layer_image.shape[0]
        cols = prev_layer_image.shape[1]
        out_rows = math.ceil(rows / float(downscale))
        out_cols = math.ceil(cols / float(downscale))

        resized = resize(prev_layer_image, (out_rows, out_cols), order=order,
                         mode=mode, cval=cval)
        layer_image = _smooth(resized, sigma, mode, cval)

        prev_rows = rows
        prev_cols = cols
        prev_layer_image = layer_image
        rows = layer_image.shape[0]
        cols = layer_image.shape[1]

        # no change to previous pyramid layer
        if prev_rows == rows and prev_cols == cols:
            break

        yield layer_image
