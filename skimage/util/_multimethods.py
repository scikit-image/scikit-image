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
    'apply_parallel',
    'compare_images',
    'crop',
    'dtype_limits',
    'img_as_bool',
    'img_as_float',
    'img_as_float32',
    'img_as_float64',
    'img_as_int',
    'img_as_ubyte',
    'img_as_uint',
    'invert',
    'label_points',
    'map_array',
    'montage',
    'random_noise',
    'regular_grid',
    'regular_seeds',
    'unique_rows',
    'view_as_blocks',
    'view_as_windows',
]


create_skimage_util = functools.partial(
    create_multimethod, domain="numpy.skimage.util"
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


def _function_array_kw_chunks_depth_mode_extraarguments_extrakeywords_kwonly_dtype_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        function,
        array,
        chunks=None,
        depth=0,
        mode=None,
        extra_arguments=(),
        extra_keywords={},
        **kwargs,
    ):
        kw_out = kwargs.copy()
        if 'dtype' in kw_out:
            kw_out['dtype'] = dispatchables[1]
        return (
            function,
            dispatchables[0],
            chunks,
            depth,
            mode,
            extra_arguments,
            extra_keywords,
        ), kw_out

    return self_method(*args, **kwargs)


def _image1_replacer(args, kwargs, dispatchables):
    def self_method(image1, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _ar_replacer(args, kwargs, dispatchables):
    def self_method(ar, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _coords_replacer(args, kwargs, dispatchables):
    def self_method(coords, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _inputarr_inputvals_outputvals_kw_out_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        input_arr, input_vals, output_vals, out=None, *args, **kwargs
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            dispatchables[2],
            dispatchables[3],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _arrin_kw_fill_replacer(args, kwargs, dispatchables):
    def self_method(arr_in, fill='mean', *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _arshape_replacer(args, kwargs, dispatchables):
    def self_method(ar_shape, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _arshape_npoints_kw_dtype_replacer(args, kwargs, dispatchables):
    def self_method(ar_shape, n_points, dtype='int', *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], n_points, dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _arrin_replacer(args, kwargs, dispatchables):
    def self_method(arr_in, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


@create_skimage_util(
    _function_array_kw_chunks_depth_mode_extraarguments_extrakeywords_kwonly_dtype_replacer
)
@all_of_type(ndarray)
@_get_docs
def apply_parallel(
    function,
    array,
    chunks=None,
    depth=0,
    mode=None,
    extra_arguments=(),
    extra_keywords={},
    *,
    dtype=None,
    compute=None,
    channel_axis=None,
    multichannel=False,
):
    return (array, Dispatchable(dtype, np.dtype))


@create_skimage_util(_image1_replacer)
@all_of_type(ndarray)
@_get_docs
def compare_images(image1, image2, method='diff', *, n_tiles=(8, 8)):
    return (image1,)


@create_skimage_util(_ar_replacer)
@all_of_type(ndarray)
@_get_docs
def crop(ar, crop_width, copy=False, order='K'):
    return (ar,)


@create_skimage_util(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def dtype_limits(image, clip_negative=False):
    return (image,)


@create_skimage_util(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def img_as_bool(image, force_copy=False):
    return (image,)


@create_skimage_util(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def img_as_float(image, force_copy=False):
    return (image,)


@create_skimage_util(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def img_as_float32(image, force_copy=False):
    return (image,)


@create_skimage_util(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def img_as_float64(image, force_copy=False):
    return (image,)


@create_skimage_util(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def img_as_int(image, force_copy=False):
    return (image,)


@create_skimage_util(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def img_as_ubyte(image, force_copy=False):
    return (image,)


@create_skimage_util(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def img_as_uint(image, force_copy=False):
    return (image,)


@create_skimage_util(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def invert(image, signed_float=False):
    return (image,)


@create_skimage_util(_coords_replacer)
@all_of_type(ndarray)
@_get_docs
def label_points(coords, output_shape):
    return (coords,)


@create_skimage_util(_inputarr_inputvals_outputvals_kw_out_replacer)
@all_of_type(ndarray)
@_get_docs
def map_array(input_arr, input_vals, output_vals, out=None):
    return (input_arr, input_vals, output_vals, _mark_output(out))


@create_skimage_util(_arrin_kw_fill_replacer)
@all_of_type(ndarray)
@_get_docs
def montage(
    arr_in,
    fill='mean',
    rescale_intensity=False,
    grid_shape=None,
    padding_width=0,
    multichannel=False,
    *,
    channel_axis=None,
):
    return (arr_in, fill)


@create_skimage_util(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
    return (image,)


@create_skimage_util(_arshape_replacer)
@all_of_type(ndarray)
@_get_docs
def regular_grid(ar_shape, n_points):
    return (ar_shape,)


@create_skimage_util(_arshape_npoints_kw_dtype_replacer)
@all_of_type(ndarray)
@_get_docs
def regular_seeds(ar_shape, n_points, dtype='int'):
    return (ar_shape, Dispatchable(dtype, np.dtype))


@create_skimage_util(_ar_replacer)
@all_of_type(ndarray)
@_get_docs
def unique_rows(ar):
    return (ar,)


@create_skimage_util(_arrin_replacer)
@all_of_type(ndarray)
@_get_docs
def view_as_blocks(arr_in, block_shape):
    return (arr_in,)


@create_skimage_util(_arrin_replacer)
@all_of_type(ndarray)
@_get_docs
def view_as_windows(arr_in, window_shape, step=1):
    return (arr_in,)
