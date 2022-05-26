import functools
import warnings

import numpy as np
from numpy import dtype, ndarray
from uarray import (
    Dispatchable,
    all_of_type,
    create_multimethod,
    generate_multimethod,
)

from skimage._backend import _mark_output, _mark_scalar_or_array

from . import _api

__all__ = [
    'adjust_gamma',
    'adjust_log',
    'adjust_sigmoid',
    'cumulative_distribution',
    'equalize_adapthist',
    'equalize_hist',
    'histogram',
    'is_low_contrast',
    'match_histograms',
    'rescale_intensity',
]


create_skimage_exposure = functools.partial(
    create_multimethod, domain="numpy.skimage.exposure"
)


def _get_docs(func):
    """
    Decorator to take the docstring from original
    function and assign to the multimethod.
    """
    func.__doc__ = getattr(_api, func.__name__).__doc__
    return func


def _identity_replacer(args, kwargs, arrays):
    return args, kwargs


def _image_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """

    def self_method(image, *args, **kwargs):
        return (dispatchables[0],) + args, kwargs

    return self_method(*args, **kwargs)


def _image_kw_kernelsize_replacer(args, kwargs, dispatchables):
    def self_method(image, kernel_size=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_nbins_mask_replacer(args, kwargs, dispatchables):
    def self_method(image, nbins=256, mask=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], nbins, dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_reference_replacer(args, kwargs, dispatchables):
    def self_method(image, reference, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


@create_skimage_exposure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def adjust_gamma(image, gamma=1, gain=1):
    return (image,)


@create_skimage_exposure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def adjust_log(image, gain=1, inv=False):
    return (image,)


@create_skimage_exposure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False):
    return (image,)


@create_skimage_exposure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def cumulative_distribution(image, nbins=256):
    return (image,)


@create_skimage_exposure(_image_kw_kernelsize_replacer)
@all_of_type(ndarray)
@_get_docs
def equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256):
    return (image, kernel_size)


@create_skimage_exposure(_image_kw_nbins_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def equalize_hist(image, nbins=256, mask=None):
    return (image, mask)


@create_skimage_exposure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def histogram(
    image,
    nbins=256,
    source_range='image',
    normalize=False,
    *,
    channel_axis=None,
):
    return (image,)


@create_skimage_exposure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def is_low_contrast(
    image,
    fraction_threshold=0.05,
    lower_percentile=1,
    upper_percentile=99,
    method='linear',
):
    return (image,)


@create_skimage_exposure(_image_reference_replacer)
@all_of_type(ndarray)
@_get_docs
def match_histograms(
    image, reference, *, channel_axis=None, multichannel=False
):
    return (image, reference)


@create_skimage_exposure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def rescale_intensity(image, in_range='image', out_range='dtype'):
    return (image,)
