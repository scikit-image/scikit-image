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
    'ahx_from_rgb',
    'bex_from_rgb',
    'bpx_from_rgb',
    'bro_from_rgb',
    'color_dict',
    'combine_stains',
    'convert_colorspace',
    'deltaE_cie76',
    'deltaE_ciede2000',
    'deltaE_ciede94',
    'deltaE_cmc',
    'fgx_from_rgb',
    'gdx_from_rgb',
    'gray2rgb',
    'gray2rgba',
    'hax_from_rgb',
    'hdx_from_rgb',
    'hed2rgb',
    'hed_from_rgb',
    'hpx_from_rgb',
    'hsv2rgb',
    'lab2lch',
    'lab2rgb',
    'lab2xyz',
    'label2rgb',
    'lch2lab',
    'rbd_from_rgb',
    'rgb2gray',
    'rgb2hed',
    'rgb2hsv',
    'rgb2lab',
    'rgb2rgbcie',
    'rgb2xyz',
    'rgb2ycbcr',
    'rgb2ydbdr',
    'rgb2yiq',
    'rgb2ypbpr',
    'rgb2yuv',
    'rgb_from_ahx',
    'rgb_from_bex',
    'rgb_from_bpx',
    'rgb_from_bro',
    'rgb_from_fgx',
    'rgb_from_gdx',
    'rgb_from_hax',
    'rgb_from_hdx',
    'rgb_from_hed',
    'rgb_from_hpx',
    'rgb_from_rbd',
    'rgba2rgb',
    'rgbcie2rgb',
    'separate_stains',
    'xyz2lab',
    'xyz2rgb',
    'ycbcr2rgb',
    'ydbdr2rgb',
    'yiq2rgb',
    'ypbpr2rgb',
    'yuv2rgb',
]


create_skimage_color = functools.partial(
    create_multimethod, domain="numpy.skimage.color"
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


def _stains_convmatrix_replacer(args, kwargs, dispatchables):
    def self_method(stains, conv_matrix, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _arr_replacer(args, kwargs, dispatchables):
    def self_method(arr, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _lab1_lab2_replacer(args, kwargs, dispatchables):
    def self_method(lab1, lab2, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_alpha_replacer(args, kwargs, dispatchables):
    def self_method(image, alpha=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _hed_replacer(args, kwargs, dispatchables):
    def self_method(hed, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _hsv_replacer(args, kwargs, dispatchables):
    def self_method(hsv, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _lab_replacer(args, kwargs, dispatchables):
    def self_method(lab, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _label_kw_image_colors_alpha_bglabel_bgcolor_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        label,
        image=None,
        colors=None,
        alpha=0.3,
        bg_label=0,
        bg_color=(0, 0, 0),
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            colors,
            dispatchables[2],
            bg_label,
            bg_color,
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _lch_replacer(args, kwargs, dispatchables):
    def self_method(lch, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _luv_replacer(args, kwargs, dispatchables):
    def self_method(luv, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _rgb_replacer(args, kwargs, dispatchables):
    def self_method(rgb, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _rgba_kw_background_replacer(args, kwargs, dispatchables):
    def self_method(rgba, background=(1, 1, 1), *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _rgbcie_replacer(args, kwargs, dispatchables):
    def self_method(rgbcie, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _rgb_convmatrix_replacer(args, kwargs, dispatchables):
    def self_method(rgb, conv_matrix, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _xyz_replacer(args, kwargs, dispatchables):
    def self_method(xyz, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _ycbcr_replacer(args, kwargs, dispatchables):
    def self_method(ycbcr, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _ydbdr_replacer(args, kwargs, dispatchables):
    def self_method(ydbdr, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _yiq_replacer(args, kwargs, dispatchables):
    def self_method(yiq, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _ypbpr_replacer(args, kwargs, dispatchables):
    def self_method(ypbpr, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _yuv_replacer(args, kwargs, dispatchables):
    def self_method(yuv, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


@create_skimage_color(_stains_convmatrix_replacer)
@all_of_type(ndarray)
@_get_docs
def combine_stains(stains, conv_matrix, *, channel_axis=-1):
    return (stains, conv_matrix)


@create_skimage_color(_arr_replacer)
@all_of_type(ndarray)
@_get_docs
def convert_colorspace(arr, fromspace, tospace, *, channel_axis=-1):
    return (arr,)


@create_skimage_color(_lab1_lab2_replacer)
@all_of_type(ndarray)
@_get_docs
def deltaE_cie76(lab1, lab2, channel_axis=-1):
    return (lab1, lab2)


@create_skimage_color(_lab1_lab2_replacer)
@all_of_type(ndarray)
@_get_docs
def deltaE_ciede2000(lab1, lab2, kL=1, kC=1, kH=1, *, channel_axis=-1):
    return (lab1, lab2)


@create_skimage_color(_lab1_lab2_replacer)
@all_of_type(ndarray)
@_get_docs
def deltaE_ciede94(
    lab1, lab2, kH=1, kC=1, kL=1, k1=0.045, k2=0.015, *, channel_axis=-1
):
    return (lab1, lab2)


@create_skimage_color(_lab1_lab2_replacer)
@all_of_type(ndarray)
@_get_docs
def deltaE_cmc(lab1, lab2, kL=1, kC=1, *, channel_axis=-1):
    return (lab1, lab2)


@create_skimage_color(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def gray2rgb(image, *, channel_axis=-1):
    return (image,)


@create_skimage_color(_image_kw_alpha_replacer)
@all_of_type(ndarray)
@_get_docs
def gray2rgba(image, alpha=None, *, channel_axis=-1):
    return (image, alpha)


@create_skimage_color(_hed_replacer)
@all_of_type(ndarray)
@_get_docs
def hed2rgb(hed, *, channel_axis=-1):
    return (hed,)


@create_skimage_color(_hsv_replacer)
@all_of_type(ndarray)
@_get_docs
def hsv2rgb(hsv, *, channel_axis=-1):
    return (hsv,)


@create_skimage_color(_lab_replacer)
@all_of_type(ndarray)
@_get_docs
def lab2lch(lab, *, channel_axis=-1):
    return (lab,)


@create_skimage_color(_lab_replacer)
@all_of_type(ndarray)
@_get_docs
def lab2rgb(lab, illuminant='D65', observer='2', *, channel_axis=-1):
    return (lab,)


@create_skimage_color(_lab_replacer)
@all_of_type(ndarray)
@_get_docs
def lab2xyz(lab, illuminant='D65', observer='2', *, channel_axis=-1):
    return (lab,)


@create_skimage_color(_label_kw_image_colors_alpha_bglabel_bgcolor_replacer)
@all_of_type(ndarray)
@_get_docs
def label2rgb(
    label,
    image=None,
    colors=None,
    alpha=0.3,
    bg_label=0,
    bg_color=(0, 0, 0),
    image_alpha=1,
    kind='overlay',
    *,
    saturation=0,
    channel_axis=-1,
):
    return (label, image, alpha)


@create_skimage_color(_lch_replacer)
@all_of_type(ndarray)
@_get_docs
def lch2lab(lch, *, channel_axis=-1):
    return (lch,)


@create_skimage_color(_luv_replacer)
@all_of_type(ndarray)
@_get_docs
def luv2rgb(luv, *, channel_axis=-1):
    return (luv,)


@create_skimage_color(_luv_replacer)
@all_of_type(ndarray)
@_get_docs
def luv2xyz(luv, illuminant='D65', observer='2', *, channel_axis=-1):
    return (luv,)


@create_skimage_color(_rgb_replacer)
@all_of_type(ndarray)
@_get_docs
def rgb2gray(rgb, *, channel_axis=-1):
    return (rgb,)


@create_skimage_color(_rgb_replacer)
@all_of_type(ndarray)
@_get_docs
def rgb2hed(rgb, *, channel_axis=-1):
    return (rgb,)


@create_skimage_color(_rgb_replacer)
@all_of_type(ndarray)
@_get_docs
def rgb2hsv(rgb, *, channel_axis=-1):
    return (rgb,)


@create_skimage_color(_rgb_replacer)
@all_of_type(ndarray)
@_get_docs
def rgb2lab(rgb, illuminant='D65', observer='2', *, channel_axis=-1):
    return (rgb,)


@create_skimage_color(_rgb_replacer)
@all_of_type(ndarray)
@_get_docs
def rgb2luv(rgb, *, channel_axis=-1):
    return (rgb,)


@create_skimage_color(_rgb_replacer)
@all_of_type(ndarray)
@_get_docs
def rgb2rgbcie(rgb, *, channel_axis=-1):
    return (rgb,)


@create_skimage_color(_rgb_replacer)
@all_of_type(ndarray)
@_get_docs
def rgb2xyz(rgb, *, channel_axis=-1):
    return (rgb,)


@create_skimage_color(_rgb_replacer)
@all_of_type(ndarray)
@_get_docs
def rgb2ycbcr(rgb, *, channel_axis=-1):
    return (rgb,)


@create_skimage_color(_rgb_replacer)
@all_of_type(ndarray)
@_get_docs
def rgb2ydbdr(rgb, *, channel_axis=-1):
    return (rgb,)


@create_skimage_color(_rgb_replacer)
@all_of_type(ndarray)
@_get_docs
def rgb2yiq(rgb, *, channel_axis=-1):
    return (rgb,)


@create_skimage_color(_rgb_replacer)
@all_of_type(ndarray)
@_get_docs
def rgb2ypbpr(rgb, *, channel_axis=-1):
    return (rgb,)


@create_skimage_color(_rgb_replacer)
@all_of_type(ndarray)
@_get_docs
def rgb2yuv(rgb, *, channel_axis=-1):
    return (rgb,)


@create_skimage_color(_rgba_kw_background_replacer)
@all_of_type(ndarray)
@_get_docs
def rgba2rgb(rgba, background=(1, 1, 1), *, channel_axis=-1):
    return (rgba, background)


@create_skimage_color(_rgbcie_replacer)
@all_of_type(ndarray)
@_get_docs
def rgbcie2rgb(rgbcie, *, channel_axis=-1):
    return (rgbcie,)


@create_skimage_color(_rgb_convmatrix_replacer)
@all_of_type(ndarray)
@_get_docs
def separate_stains(rgb, conv_matrix, *, channel_axis=-1):
    return (rgb, conv_matrix)


@create_skimage_color(_xyz_replacer)
@all_of_type(ndarray)
@_get_docs
def xyz2lab(xyz, illuminant='D65', observer='2', *, channel_axis=-1):
    return (xyz,)


@create_skimage_color(_xyz_replacer)
@all_of_type(ndarray)
@_get_docs
def xyz2luv(xyz, illuminant='D65', observer='2', *, channel_axis=-1):
    return (xyz,)


@create_skimage_color(_xyz_replacer)
@all_of_type(ndarray)
@_get_docs
def xyz2rgb(xyz, *, channel_axis=-1):
    return (xyz,)


@create_skimage_color(_ycbcr_replacer)
@all_of_type(ndarray)
@_get_docs
def ycbcr2rgb(ycbcr, *, channel_axis=-1):
    return (ycbcr,)


@create_skimage_color(_ydbdr_replacer)
@all_of_type(ndarray)
@_get_docs
def ydbdr2rgb(ydbdr, *, channel_axis=-1):
    return (ydbdr,)


@create_skimage_color(_yiq_replacer)
@all_of_type(ndarray)
@_get_docs
def yiq2rgb(yiq, *, channel_axis=-1):
    return (yiq,)


@create_skimage_color(_ypbpr_replacer)
@all_of_type(ndarray)
@_get_docs
def ypbpr2rgb(ypbpr, *, channel_axis=-1):
    return (ypbpr,)


@create_skimage_color(_yuv_replacer)
@all_of_type(ndarray)
@_get_docs
def yuv2rgb(yuv, *, channel_axis=-1):
    return (yuv,)
