import functools
import warnings

import numpy as np
from numpy import dtype, ndarray
from uarray import generate_multimethod, Dispatchable
from uarray import all_of_type, create_multimethod

from .._backend import _mark_output, _mark_scalar_or_array
from .._shared.multimethods import gaussian
from . import _api

__all__ = [
    "gaussian",
    "difference_of_gaussians",
    "gabor",
    "gabor_kernel",
    "median",
    "rank_order",
    "correlate_sparse",
    "unsharp_mask",
    "window",
    "sobel",
    "sobel_h",
    "sobel_v",
    "scharr",
    "scharr_h",
    "scharr_v",
    "prewitt",
    "prewitt_h",
    "prewitt_v",
    "roberts",
    "roberts_pos_diag",
    "roberts_neg_diag",
    "laplace",
    "farid",
    "farid_h",
    "farid_v",
    "forward",
    "inverse",
    "wiener",
    "compute_hessian_eigenvalues",
    "meijering",
    "sato",
    "frangi",
    "hessian",
    "try_all_threshold",
    "threshold_local",
    "threshold_otsu",
    "threshold_yen",
    "threshold_isodata",
    "threshold_li",
    "threshold_minimum",
    "threshold_mean",
    "threshold_triangle",
    "threshold_niblack",
    "threshold_sauvola",
    "apply_hysteresis_threshold",
    "threshold_multiotsu",
]

create_skimage_filters = functools.partial(
    create_multimethod, domain="numpy.skimage.filters"
)


def _get_docs(func):
    """
    Decorator to take the docstring from original
    function and assign to the multimethod.
    """
    func.__doc__ = getattr(_api, func.__name__).__doc__
    return func


def _identity_arg_replacer(args, kwargs, arrays):
    return args, kwargs


def _image_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """

    def self_method(image, *args, **kwargs):
        return (dispatchables[0],) + args, kwargs

    return self_method(*args, **kwargs)


def _image_kernel_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """

    def self_method(image, kernel, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


def _image_low_high_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """

    def self_method(image, low, high, *args, **kwargs):
        return tuple(dispatchables[:3]) + args, kwargs

    return self_method(*args, **kwargs)


""" _fft_based.py multimethods """


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def butterworth(
    image, cutoff_frequency_ratio=0.005, high_pass=True, order=2.0, channel_axis=None
):
    return (image,)


""" _gaussian.py multimethods """

# Note: gaussian multimethod is imported from _shared._multimethods

@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def difference_of_gaussians(
    image,
    low_sigma,
    high_sigma=None,
    *,
    mode="nearest",
    cval=0,
    channel_axis=None,
    truncate=4.0
):
    return (image,)


""" _gabor.py multimethods """


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def gabor(
    image,
    frequency,
    theta=0,
    bandwidth=1,
    sigma_x=None,
    sigma_y=None,
    n_stds=3,
    offset=0,
    mode="reflect",
    cval=0,
):
    return (image,)


@create_skimage_filters(_identity_arg_replacer)
@all_of_type(dtype)
@_get_docs
def gabor_kernel(
    frequency,
    theta=0,
    bandwidth=1,
    sigma_x=None,
    sigma_y=None,
    n_stds=3,
    offset=0,
    dtype=np.complex128,
):
    return (dtype,)


""" _median.py multimethods """

# create_skimage_filters must be the outermost decorator
def _image_footprint_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """

    def self_method(image, *args, **kwargs):
        kw_out = kwargs.copy()
        if "footprint" in kwargs:
            kw_out["footprint"] = dispatchables[1]
        return (dispatchables[0],) + args, kwargs

    return self_method(*args, **kwargs)


# TODO: had to add **kwargs and handle deprecated selem manually!
#       wrapping with the multimethod breaks deprecate_kwarg decorator use.
#       if deprecate_kwarg is used above @create_skimage_filters below, the
#       deprecation works, but then the multimethod is found by the backends!


@create_skimage_filters(_image_footprint_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def median(
    image, footprint=None, out=None, mode="nearest", cval=0.0, behavior="ndimage"
):
    return (image, footprint, _mark_output(out))


""" _rank_order.py multimethods """


@create_skimage_filters(_image_arg_replacer)
@_get_docs
def rank_order(image):
    return


""" _sparse.py multimethods """


@create_skimage_filters(_image_kernel_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def correlate_sparse(image, kernel, mode="reflect"):
    return (image, kernel)


""" _unsharp_mask.py multimethods """


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def unsharp_mask(
    image,
    radius=1.0,
    amount=1.0,
    preserve_range=False,
    *,
    channel_axis=None
):
    return (image,)


""" _window.py multimethods """


@create_skimage_filters(_identity_arg_replacer)
@_get_docs
def window(window_type, shape, warp_kwargs=None):
    return ()


""" edges.py multimethods """


def _image_mask_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """

    def self_method(image, mask=None, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def sobel(image, mask=None, *, axis=None, mode="reflect", cval=0.0):
    return (image, mask)


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
def sobel_h(image, mask=None):
    return (image, mask)


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def sobel_v(image, mask=None):
    return (image, mask)


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def scharr(image, mask=None, *, axis=None, mode="reflect", cval=0.0):
    return (image, mask)


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def scharr_h(image, mask=None):
    return (image, mask)


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def scharr_v(image, mask=None):
    return (image, mask)


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def prewitt(image, mask=None, *, axis=None, mode="reflect", cval=0.0):
    return (image, mask)


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def prewitt_h(image, mask=None):
    return (image, mask)


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def prewitt_v(image, mask=None):
    return (image, mask)


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def roberts(image, mask=None):
    return (image, mask)


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def roberts_pos_diag(image, mask=None):
    return (image, mask)


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def roberts_neg_diag(image, mask=None):
    return (image, mask)


def _image_ksize_mask_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """

    def self_method(image, ksize=3, mask=None, *args, **kwargs):
        return (dispatchables[0], ksize, dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@create_skimage_filters(_image_ksize_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def laplace(image, ksize=3, mask=None):
    return (image, mask)


def _image_kwarg_mask_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """

    def self_method(image, *args, **kwargs):
        kwargs_out = kwargs.copy()
        kwargs_out["mask"] = dispatchables[1]
        return (dispatchables[0],) + args, kwargs

    return self_method(*args, **kwargs)


@create_skimage_filters(_image_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def farid(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    return (image, mask)


@create_skimage_filters(_image_kwarg_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def farid_h(image, *, mask=None):
    return (image, mask)


@create_skimage_filters(_image_kwarg_mask_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def farid_v(image, *, mask=None):
    return (image, mask)


""" lpi_filter.py multimethods """


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def forward(data, impulse_response=None, filter_params={}, predefined_filter=None):
    return (data,)


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def inverse(
    data, impulse_response=None, filter_params={}, max_gain=2, predefined_filter=None
):
    return (data,)


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def wiener(
    data, impulse_response=None, filter_params={}, K=0.25, predefined_filter=None
):
    return (data,)


""" ridges.py multimethods """


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def compute_hessian_eigenvalues(image, sigma, sorting="none", mode="constant", cval=0):
    return (image,)


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def meijering(
    image, sigmas=range(1, 10, 2), alpha=None, black_ridges=True, mode="reflect", cval=0
):
    return (image,)


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def sato(image, sigmas=range(1, 10, 2), black_ridges=True, mode=None, cval=0):
    return (image,)


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def frangi(
    image,
    sigmas=range(1, 10, 2),
    scale_range=None,
    scale_step=None,
    alpha=0.5,
    beta=0.5,
    gamma=15,
    black_ridges=True,
    mode="reflect",
    cval=0,
):
    return (image,)


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def hessian(
    image,
    sigmas=range(1, 10, 2),
    scale_range=None,
    scale_step=None,
    alpha=0.5,
    beta=0.5,
    gamma=15,
    black_ridges=True,
    mode=None,
    cval=0,
):
    return (image,)


""" thresholding.py multimethods """


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def try_all_threshold(image, figsize=(8, 5), verbose=True):
    return (image,)


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def threshold_local(
    image, block_size, method="gaussian", offset=0, mode="reflect", param=None, cval=0
):
    return (image,)


def _image_kwarg_hist_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """

    def self_method(image, *args, **kwargs):
        kw_out = kwargs.copy()
        if "hist" in kwargs:
            kw_out["hist"] = dispatchables[1]
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def kwarg_image_and_hist_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """

    def self_method(*args, **kwargs):
        kw_out = kwargs.copy()
        if "image" in kwargs:
            kw_out["image"] = dispatchables[0]
        if "hist" in kwargs:
            kw_out["hist"] = dispatchables[1]
        return args, kw_out

    return self_method(*args, **kwargs)


@create_skimage_filters(kwarg_image_and_hist_replacer)
@all_of_type(ndarray)
@_get_docs
def threshold_otsu(image=None, nbins=256, *, hist=None):
    return (image, hist)


@create_skimage_filters(kwarg_image_and_hist_replacer)
@all_of_type(ndarray)
@_get_docs
def threshold_yen(image=None, nbins=256, *, hist=None):
    return (image, hist)


@create_skimage_filters(kwarg_image_and_hist_replacer)
@all_of_type(ndarray)
@_get_docs
def threshold_isodata(image=None, nbins=256, return_all=False, *, hist=None):
    return (image, hist)


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def threshold_li(image, *, tolerance=None, initial_guess=None, iter_callback=None):
    return (image,)


@create_skimage_filters(kwarg_image_and_hist_replacer)
@all_of_type(ndarray)
@_get_docs
def threshold_minimum(image=None, nbins=256, max_num_iter=10000, *, hist=None):
    return (image, hist)


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def threshold_mean(image):
    return (image,)


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def threshold_triangle(image, nbins=256):
    return (image,)


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def threshold_niblack(image, window_size=15, k=0.2):
    return (image,)


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def threshold_sauvola(image, window_size=15, k=0.2, r=None):
    return (image,)


# TODO: low, high can be float or ndarray
@create_skimage_filters(_image_low_high_arg_replacer)
@all_of_type(ndarray)
@_get_docs
def apply_hysteresis_threshold(image, low, high):
    return (image, _mark_scalar_or_array(low), _mark_scalar_or_array(high))


@create_skimage_filters(kwarg_image_and_hist_replacer)
@all_of_type(ndarray)
@_get_docs
def threshold_multiotsu(image=None, classes=3, nbins=256, hist=None):
    return (image, hist)
