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
from unumpy import mark_dtype

from skimage._backend import _mark_output, _mark_scalar_or_array

from . import _api

__all__ = [
    'adapted_rand_error',
    'contingency_table',
    'hausdorff_distance',
    'hausdorff_pair',
    'mean_squared_error',
    'normalized_mutual_information',
    'normalized_root_mse',
    'peak_signal_noise_ratio',
    'structural_similarity',
    'variation_of_information',
]


create_skimage_metrics = functools.partial(
    create_multimethod, domain="numpy.skimage.metrics"
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


def __kw_imagetrue_imagetest_kwonly_table_replacer(
    args, kwargs, dispatchables
):
    def self_method(image_true=None, image_test=None, **kwargs):
        kw_out = kwargs.copy()
        if 'table' in kw_out:
            kw_out['table'] = dispatchables[2]
        return (dispatchables[0], dispatchables[1]), kw_out

    return self_method(*args, **kwargs)


def _imtrue_imtest_replacer(args, kwargs, dispatchables):
    def self_method(im_true, im_test, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image0_image1_replacer(args, kwargs, dispatchables):
    def self_method(image0, image1, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _imagetrue_imagetest_replacer(args, kwargs, dispatchables):
    def self_method(image_true, image_test, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _im1_im2_replacer(args, kwargs, dispatchables):
    def self_method(im1, im2, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def __kw_image0_image1_kwonly_table_replacer(args, kwargs, dispatchables):
    def self_method(image0=None, image1=None, **kwargs):
        kw_out = kwargs.copy()
        if 'table' in kw_out:
            kw_out['table'] = dispatchables[2]
        return (dispatchables[0], dispatchables[1]), kw_out

    return self_method(*args, **kwargs)


@create_skimage_metrics(__kw_imagetrue_imagetest_kwonly_table_replacer)
@all_of_type(ndarray)
@_get_docs
def adapted_rand_error(
    image_true=None, image_test=None, *, table=None, ignore_labels=(0,)
):
    return (image_true, image_test, table)


@create_skimage_metrics(_imtrue_imtest_replacer)
@all_of_type(ndarray)
@_get_docs
def contingency_table(
    im_true, im_test, *, ignore_labels=None, normalize=False
):
    return (im_true, im_test)


@create_skimage_metrics(_image0_image1_replacer)
@all_of_type(ndarray)
@_get_docs
def hausdorff_distance(image0, image1, method="standard"):
    return (image0, image1)


@create_skimage_metrics(_image0_image1_replacer)
@all_of_type(ndarray)
@_get_docs
def hausdorff_pair(image0, image1):
    return (image0, image1)


@create_skimage_metrics(_image0_image1_replacer)
@all_of_type(ndarray)
@_get_docs
def mean_squared_error(image0, image1):
    return (image0, image1)


@create_skimage_metrics(_image0_image1_replacer)
@all_of_type(ndarray)
@_get_docs
def normalized_mutual_information(image0, image1, *, bins=100):
    return (image0, image1)


@create_skimage_metrics(_imagetrue_imagetest_replacer)
@all_of_type(ndarray)
@_get_docs
def normalized_root_mse(image_true, image_test, *, normalization='euclidean'):
    return (image_true, image_test)


@create_skimage_metrics(_imagetrue_imagetest_replacer)
@all_of_type(ndarray)
@_get_docs
def peak_signal_noise_ratio(image_true, image_test, *, data_range=None):
    return (image_true, image_test)


@create_skimage_metrics(_im1_im2_replacer)
@all_of_type(ndarray)
@_get_docs
def structural_similarity(
    im1,
    im2,
    *,
    win_size=None,
    gradient=False,
    data_range=None,
    channel_axis=None,
    multichannel=False,
    gaussian_weights=False,
    full=False,
    **kwargs,
):
    return (im1, im2)


@create_skimage_metrics(__kw_image0_image1_kwonly_table_replacer)
@all_of_type(ndarray)
@_get_docs
def variation_of_information(
    image0=None, image1=None, *, table=None, ignore_labels=()
):
    return (image0, image1, table)
