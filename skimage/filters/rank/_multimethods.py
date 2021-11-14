import functools
import warnings

import numpy as np
from numpy import dtype, ndarray
from uarray import generate_multimethod, Dispatchable
from uarray import all_of_type, create_multimethod
from unumpy import mark_dtype

from . import _api
from skimage._backend import scalar_or_array

__all__ = [
    'autolevel',
    'autolevel_percentile',
    'enhance_contrast',
    'enhance_contrast_percentile',
    'entropy',
    'equalize',
    'geometric_mean',
    'gradient',
    'gradient_percentile',
    'majority',
    'maximum',
    'mean',
    'mean_bilateral',
    'mean_percentile',
    'median',
    'minimum',
    'modal',
    'noise_filter',
    'otsu',
    'percentile',
    'pop',
    'pop_bilateral',
    'pop_percentile',
    'subtract_mean',
    'subtract_mean_percentile',
    'sum',
    'sum_bilateral',
    'sum_percentile',
    'threshold',
    'threshold_percentile',
    'windowed_histogram',
]


create_skimage_rank = functools.partial(
    create_multimethod, domain="numpy.skimage.filters.rank"
)


_mark_scalar_or_array = functools.partial(
    Dispatchable, dispatch_type=scalar_or_array, coercible=True
)

_mark_non_coercible = functools.partial(
    Dispatchable, dispatch_type=np.ndarray, coercible=False
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


def _image_footprint_kw_out_mask_replacer(args, kwargs, dispatchables):
    def self_method(image, footprint, out=None, mask=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            dispatchables[2],
            dispatchables[3],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_footprint_kwonly_out_mask_replacer(args, kwargs, dispatchables):
    def self_method(image, footprint, **kwargs):
        kw_out = kwargs.copy()
        if 'out' in kw_out:
            kw_out['out'] = dispatchables[2]
        if 'mask' in kw_out:
            kw_out['mask'] = dispatchables[3]
        return (dispatchables[0], dispatchables[1]), kw_out

    return self_method(*args, **kwargs)


def _image_kw_footprint_out_mask_replacer(args, kwargs, dispatchables):
    def self_method(
        image, footprint=None, out=None, mask=None, *args, **kwargs
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            dispatchables[2],
            dispatchables[3],
        ) + args, kw_out

    return self_method(*args, **kwargs)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def autolevel(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def autolevel_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    p0=0,
    p1=1,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def enhance_contrast(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def enhance_contrast_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    p0=0,
    p1=1,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def entropy(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def equalize(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def geometric_mean(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def gradient(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def gradient_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    p0=0,
    p1=1,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kwonly_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def majority(
    image,
    footprint,
    *,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def maximum(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def mean(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def mean_bilateral(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    s0=10,
    s1=10,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def mean_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    p0=0,
    p1=1,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_kw_footprint_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def median(
    image,
    footprint=None,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def minimum(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def modal(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def noise_filter(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def otsu(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def percentile(
    image, footprint, out=None, mask=None, shift_x=False, shift_y=False, p0=0
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def pop(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def pop_bilateral(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    s0=10,
    s1=10,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def pop_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    p0=0,
    p1=1,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def subtract_mean(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def subtract_mean_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    p0=0,
    p1=1,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def sum(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def sum_bilateral(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    s0=10,
    s1=10,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def sum_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    p0=0,
    p1=1,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def threshold(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    shift_z=False,
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def threshold_percentile(
    image, footprint, out=None, mask=None, shift_x=False, shift_y=False, p0=0
):
    return (image, footprint, _mark_non_coercible(out), mask)


@create_skimage_rank(_image_footprint_kw_out_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def windowed_histogram(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=False,
    shift_y=False,
    n_bins=None,
):
    return (image, footprint, _mark_non_coercible(out), mask)
