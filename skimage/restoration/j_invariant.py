import numpy as np
from itertools import product
from scipy import ndimage as ndi
from ..measure import compare_mse
from ..util import img_as_float


def interpolate_image(image, multichannel=False):
    spatialdims = image.ndim if not multichannel else image.ndim - 1
    conv_filter = ndi.generate_binary_structure(spatialdims, 1).astype(float)
    conv_filter.ravel()[conv_filter.size // 2] = 0
    conv_filter /= conv_filter.sum()

    if multichannel:
        interp = np.zeros(image.shape)
        for i in range(image.shape[-1]):
            interp[..., i] = ndi.convolve(image[..., i], conv_filter, mode='reflect')
    else:
        interp = ndi.convolve(image, conv_filter, mode='reflect')
    return interp


def generate_mask(shape, idx, stride=3):
    phases = np.unravel_index(idx, (stride,) * len(shape))
    mask = tuple(slice(p, None, stride) for p in phases)

    return mask


def invariant_denoise(image, denoise_function, *, stride=4, multichannel=False, masks=None, denoiser_kwargs=None):
    image = img_as_float(image)
    if denoiser_kwargs is None:
        denoiser_kwargs = {}

    interp = interpolate_image(image, multichannel=multichannel)
    output = np.zeros(image.shape)

    if masks is None:
        spatialdims = image.ndim if not multichannel else image.ndim - 1
        n_masks = stride ** spatialdims
        masks = (generate_mask(image.shape[:spatialdims], idx, stride=stride) for idx in range(n_masks))

    for mask in masks:
        input_image = image
        input_image[mask] = interp[mask]
        output[mask] = denoise_function(input_image, **denoiser_kwargs)[mask]
    return output


def selections_from_dict(dictionary):
    keys = dictionary.keys()
    for element in product(*dictionary.values()):
        yield dict(zip(keys, element))


def calibrate_denoiser(image, denoise_function, parameter_ranges, *,
                       stride=4, multichannel=False, approximate_loss=True):
    image = img_as_float(image)
    parameters_tested = list(selections_from_dict(parameter_ranges))
    losses = []

    for denoiser_kwargs in parameters_tested:
        if not approximate_loss:
            denoised = invariant_denoise(image, denoise_function, 
                                         stride=stride, multichannel=multichannel,
                                         denoiser_kwargs=denoiser_kwargs)
            loss = compare_mse(denoised, image)
        else:
            spatialdims = image.ndim if not multichannel else image.ndim - 1
            n_masks = stride ** spatialdims
            mask = generate_mask(image.shape[:spatialdims], n_masks//2, stride=stride)

            masked_denoised = invariant_denoise(image, denoise_function,
                                         masks=[mask], multichannel=multichannel,
                                         denoiser_kwargs=denoiser_kwargs)

            loss = compare_mse(masked_denoised[mask], image[mask])

        losses.append(loss)

    idx = np.argmin(losses)
    best_loss = losses[idx]
    best_parameters = parameters_tested[idx]

    denoised = invariant_denoise(image, denoise_function,
                                 stride=stride, multichannel=multichannel,
                                 denoiser_kwargs=best_parameters)

    return denoised, best_parameters, best_loss, parameters_tested, losses
