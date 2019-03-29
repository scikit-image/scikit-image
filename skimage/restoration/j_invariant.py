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


def generate_mask(shape, idx, grid_width=3):
    m = np.zeros(shape)

    phasex = idx % grid_width
    phasey = (idx // grid_width) % grid_width

    m[phasex::grid_width, phasey::grid_width] = 1
    return m


def invariant_denoise(image, denoise_function, denoiser_kwargs, grid_width):
    n_masks = grid_width*grid_width

    interp = interpolate_image(image)

    output = np.zeros(image.shape)

    if denoiser_kwargs is None:
        denoiser_kwargs = {}

    for i in range(n_masks):
        m = generate_mask(image.shape, i, grid_width=grid_width)
        input_image = m*interp + (1 - m)*image
        input_image = input_image.astype(image.dtype)
        output += m*denoise_function(input_image, **denoiser_kwargs)
    return output


def selections_from_dict(dictionary):
    keys = dictionary.keys()
    for element in product(*dictionary.values()):
        yield dict(zip(keys, element))


def calibrate_denoiser(image, denoising_function, parameter_ranges,
                       grid_width=4, return_invariant=False, full_loss=False):
    image = img_as_float(image)
    parameters_tested = list(selections_from_dict(parameter_ranges))
    losses = []

    for parameters in parameters_tested:
        if full_loss:
            denoised = invariant_denoise(image, denoising_function, parameters, grid_width)
            loss = compare_mse(denoised, image)
        else:
            n_masks = grid_width*grid_width

            m = generate_mask(image.shape, n_masks//2, grid_width=grid_width)
            interp = interpolate_image(image)

            input_image = m*interp + (1 - m)*image
            input_image = input_image.astype(image.dtype)
            masked_denoised = m*denoising_function(input_image, **parameters)
            loss = n_masks * compare_mse(masked_denoised, m*image)

        losses.append(loss)

    idx = np.argmin(losses)
    best_loss = losses[idx]
    best_parameters = parameters_tested[idx]

    if return_invariant:
        denoised = invariant_denoise(image, denoising_function, best_parameters, grid_width)
    else:
        denoised = denoising_function(image, **best_parameters)

    return denoised, best_parameters, best_loss, parameters_tested, losses
