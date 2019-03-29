import numpy as np
from itertools import product
from scipy.signal import convolve2d
from ..measure import compare_mse
from ..util import img_as_float


def interpolate_image(x, conv_filter=None):
    if conv_filter is None:
        conv_filter = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    if len(x.shape) == 2:
        return convolve2d(x, conv_filter, mode = 'same')
    else:
        assert (len(x.shape) == 3) & (x.shape[2] == 3)
        x_interp = np.zeros(x.shape)
        for i in range(3):
            x_interp[:,:,i] = convolve2d(x[:,:,i], conv_filter, mode = 'same')
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
