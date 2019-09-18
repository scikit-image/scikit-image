from itertools import product

import numpy as np
from scipy import ndimage as ndi

from ..measure import compare_mse
from ..util import img_as_float


def _interpolate_image(image, multichannel=False):
    """
    Interpolate `image`, replacing each pixel with the average of its neighbors.

    Parameters
    ----------
    image : ndarray
        Input data to be interpolated.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.

    Returns
    -------
    interp : ndarray
        Interpolated version of `image`.
    """
    spatialdims = image.ndim if not multichannel else image.ndim - 1
    conv_filter = ndi.generate_binary_structure(spatialdims, 1).astype(float)
    conv_filter.ravel()[conv_filter.size // 2] = 0
    conv_filter /= conv_filter.sum()

    if multichannel:
        interp = np.zeros(image.shape)
        for i in range(image.shape[-1]):
            interp[..., i] = ndi.convolve(image[..., i], conv_filter,
                                          mode='mirror')
    else:
        interp = ndi.convolve(image, conv_filter, mode='mirror')
    return interp


def _generate_mask(shape, idx, stride=3):
    """
    Generate a uniformly gridded boolean mask of shape `shape`,
    containing ones separated by `stride` along each dimension, with offset
    controlled by `idx`.

    Parameters
    ----------
    shape : tuple of int
        Shape of the mask.
    idx : int
        The offset of the grid of ones. Iterating over `idx` will cover the
        entire array.
    stride : int
        The spacing between ones, used in each dimension.

    Returns
    -------
    mask : ndarray
        The mask.
    """
    phases = np.unravel_index(idx, (stride,) * len(shape))
    mask = tuple(slice(p, None, stride) for p in phases)

    return mask


def invariant_denoise(image, denoise_function, *, stride=4, multichannel=False,
                      masks=None, denoiser_kwargs=None):
    """
    Apply a J-invariant version of `denoise_function`, generated with an
    iterated masking and interpolation procedure.

    Parameters
    ----------
    image : ndarray
        Input data to be denoised (converted using `img_as_float`).
    denoise_function : function
        Original denoising function.
    stride : int, optional
        Stride used in masking procedure that converts `denoise_function`
        to J-invariance.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.
    masks : list of ndarray, optional
        Set of masks to use for computing J-invariant output. If `None`,
        a full set of masks covering the image will be used.
    denoiser_kwargs:
        Keyword arguments passed to `denoise_function`.

    Returns
    -------
    best_denoise_function : function
        The optimal J-invariant version of `denoise_function`.
    output : ndarray
        Denoised image, of same shape as `image`.
    """
    image = img_as_float(image)
    if denoiser_kwargs is None:
        denoiser_kwargs = {}

    interp = _interpolate_image(image, multichannel=multichannel)
    output = np.zeros(image.shape)

    if masks is None:
        spatialdims = image.ndim if not multichannel else image.ndim - 1
        n_masks = stride ** spatialdims
        masks = (_generate_mask(image.shape[:spatialdims], idx, stride=stride)
                 for idx in range(n_masks))

    for mask in masks:
        input_image = image.copy()
        #input_image = image
        input_image[mask] = interp[mask]
        output[mask] = denoise_function(input_image, **denoiser_kwargs)[mask]
    return output


def _product_from_dict(dictionary):
    """
    Convert a dict of lists into a list of dicts whose values consist of the
    cartesian product of the values in the original dict.

    Parameters
    ----------
    dictionary : dict of lists
        Dictionary of lists to be multiplied.

    Returns
    -------
    selections : list of dicts
        List of dicts representing the cartesian product of the values
        in `dictionary`.
    """
    keys = dictionary.keys()
    for element in product(*dictionary.values()):
        yield dict(zip(keys, element))


def calibrate_denoiser(image, denoise_function, parameter_ranges, *,
                       stride=4, multichannel=False, approximate_loss=True):
    """
    Calibrate a denoising function, yielding a J-invariant version
    of `denoise_function` with the optimal parameter values for denoising
    the input image.

    Parameters
    ----------
    image : ndarray
        Input data to be denoised (converted using `img_as_float`).
    denoise_function : function
        Denoising function to be calibrated.
    parameter_ranges : dict of list
        Ranges of parameters for `denoise_function` to be calibrated over.
    stride : int, optional
        Stride used in masking procedure that converts `denoise_function`
        to J-invariance.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.
    approximate_loss : bool, optional
        Whether to approximate the self-supervised loss used to evaluate the
        denoiser by only computing it on one masked version of the image.
        If False, the runtime will be a factor of `stride**image.ndim` longer.

    Returns
    -------
    best_denoise_function : function
        The optimal J-invariant version of `denoise_function`.

    Notes
    -----

    The calibration procedure uses a self-supervised mean-square-error loss
    to evaluate the performance of J-invariant versions of `denoise_function`.
    The minimizer of the self-supervised loss is also the minimizer of the
    ground-truth loss (i.e., the true MSE error) [1]. The returned function
    can be used on the original noisy image, or other images with similar
    characteristics.

    Increasing the stride increases the performance of `best_denoise_function`
     at the expense of increasing its runtime. It has no effect on the runtime
     of the calibration.

    References
    ----------
    .. [1] J. Batson & L. Royer. Blind Denoising by Self-Supervison.
           arxiv:1901.11365.

    Examples
    --------

    >>> from skimage import color, data
    >>> from skimage.restoration import denoise_wavelet
    >>> import numpy as np
    >>> img = color.rgb2gray(data.astronaut()[:50, :50])
    >>> noisy = img + 0.5 * img.std() * np.random.randn(*img.shape)
    >>> parameter_ranges = {'sigma': np.arange(0.1, 0.4, 0.02)}
    >>> denoising_function = calibrate_denoiser(noisy, denoise_wavelet, parameter_ranges)
    >>> denoised_img = denoising_function(img)

    """
    parameters_tested, losses = calibrate_denoiser_search(image,
                                                          denoise_function,
                                                          parameter_ranges,
                                                          stride=stride,
                                                          multichannel=multichannel,
                                                          approximate_loss=approximate_loss)

    idx = np.argmin(losses)
    best_parameters = parameters_tested[idx]

    best_denoise_function = lambda x: invariant_denoise(x,
                                                        denoise_function,
                                                        stride=stride,
                                                        multichannel=multichannel,
                                                        denoiser_kwargs=best_parameters)

    return best_denoise_function


def calibrate_denoiser_search(image, denoise_function, parameter_ranges, *,
                              stride=4, multichannel=False, approximate_loss=True):
    """
    Calibrate a denoising function, yielding a J-invariant version
    of `denoise_function` with the optimal parameter values for denoising
    the input image.

    Parameters
    ----------
    image : ndarray
        Input data to be denoised (converted using `img_as_float`).
    denoise_function : function
        Denoising function to be calibrated.
    parameter_ranges : dict of list
        Ranges of parameters for `denoise_function` to be calibrated over.
    stride : int, optional
        Stride used in masking procedure that converts `denoise_function`
        to J-invariance.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.
    approximate_loss : bool, optional
        Whether to approximate the self-supervised loss used to evaluate the
        denoiser by only computing it on one masked version of the image.
        If False, the runtime will be a factor of `stride**image.ndim` longer.

    Returns
    -------
    parameters_tested : list of dict
        List of parameters tested for `denoise_function`, as a dictionary of
        kwargs.
    losses : list of int
        Self-supervised loss for each set of parameters in `parameters_tested`.
    """
    image = img_as_float(image)
    parameters_tested = list(_product_from_dict(parameter_ranges))
    losses = []

    for denoiser_kwargs in parameters_tested:
        if not approximate_loss:
            denoised = invariant_denoise(image,
                                         denoise_function,
                                         stride=stride,
                                         multichannel=multichannel,
                                         denoiser_kwargs=denoiser_kwargs)
            loss = compare_mse(denoised, image)
        else:
            spatialdims = image.ndim if not multichannel else image.ndim - 1
            n_masks = stride ** spatialdims
            mask = _generate_mask(image.shape[:spatialdims], n_masks // 2,
                                  stride=stride)

            masked_denoised = invariant_denoise(image,
                                                denoise_function,
                                                masks=[mask],
                                                multichannel=multichannel,
                                                denoiser_kwargs=denoiser_kwargs)

            loss = compare_mse(masked_denoised[mask], image[mask])

        losses.append(loss)

    return parameters_tested, losses
