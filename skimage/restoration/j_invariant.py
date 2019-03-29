import numpy as np
from itertools import product
from scipy import ndimage as ndi
from ..measure import compare_mse
from ..util import img_as_float


def interpolate_image(x, multichannel=False):
    spatialdims = x.ndim if not multichannel else x.ndim - 1
    conv_filter = ndi.generate_binary_structure(spatialdims, 1)
    conv_filter.ravel()[conv_filter.size // 2] = 0
    conv_filter /= conv_filter.sum()

    if multichannel:
        x_interp = np.zeros(x.shape)
        for i in range(x.shape[-1]):
            x_interp[..., i] = ndi.convolve(x[..., i], conv_filter, mode='reflect')
    else:
        x_interp = ndi.convolve(x, conv_filter, mode='reflect')
    return x_interp


def generate_mask(shape, idx, stride=3):
    m = np.zeros(shape)

    phasex = idx % stride
    phasey = (idx // stride) % stride

    m[phasex::stride, phasey::stride] = 1
    return m


def invariant_denoise(image, denoise_function, denoiser_kwargs, stride):
    n_masks = stride * stride

    interp = interpolate_image(image)

    output = np.zeros(image.shape)

    if denoiser_kwargs is None:
        denoiser_kwargs = {}

    for i in range(n_masks):
        m = generate_mask(image.shape, i, stride=stride)
        input_image = m*interp + (1 - m)*image
        input_image = input_image.astype(image.dtype)
        output += m*denoise_function(input_image, **denoiser_kwargs)
    return output


def selections_from_dict(dictionary):
    keys = dictionary.keys()
    for element in product(*dictionary.values()):
        yield dict(zip(keys, element))


def calibrate_denoiser(image, denoising_function, parameter_ranges,
                       stride=4, full_loss=False):
    image = img_as_float(image)
    parameters_tested = list(selections_from_dict(parameter_ranges))
    losses = []

    for parameters in parameters_tested:
        if full_loss:
            denoised = invariant_denoise(image, denoising_function, parameters, stride)
            loss = compare_mse(denoised, image)
        else:
            n_masks = stride * stride

            m = generate_mask(image.shape, n_masks // 2, stride=stride)
            interp = interpolate_image(image)

            input_image = m*interp + (1 - m)*image
            input_image = input_image.astype(image.dtype)
            masked_denoised = m*denoising_function(input_image, **parameters)
            loss = n_masks * compare_mse(masked_denoised, m*image)

        losses.append(loss)

    idx = np.argmin(losses)
    best_loss = losses[idx]
    best_parameters = parameters_tested[idx]

    denoised = invariant_denoise(image, denoising_function, best_parameters, stride)

    return denoised, best_parameters, best_loss, parameters_tested, losses
