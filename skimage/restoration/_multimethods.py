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
    'ball_kernel',
    'calibrate_denoiser',
    'cycle_spin',
    'denoise_bilateral',
    'denoise_nl_means',
    'denoise_tv_bregman',
    'denoise_tv_chambolle',
    'denoise_wavelet',
    'ellipsoid_kernel',
    'estimate_sigma',
    'inpaint_biharmonic',
    'richardson_lucy',
    'rolling_ball',
    'unsupervised_wiener',
    'unwrap_phase',
    'wiener',
]


create_skimage_restoration = functools.partial(
    create_multimethod, domain="numpy.skimage.restoration"
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


def _x_replacer(args, kwargs, dispatchables):
    def self_method(x, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _shape_replacer(args, kwargs, dispatchables):
    def self_method(shape, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _image_mask_replacer(args, kwargs, dispatchables):
    def self_method(image, mask, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_psf_replacer(args, kwargs, dispatchables):
    def self_method(image, psf, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kwonly_radius_kernel_replacer(args, kwargs, dispatchables):
    def self_method(image, **kwargs):
        kw_out = kwargs.copy()
        if 'kernel' in kw_out:
            kw_out['kernel'] = dispatchables[1]
        return (dispatchables[0],), kw_out

    return self_method(*args, **kwargs)


def _image_psf_kw_reg_replacer(args, kwargs, dispatchables):
    def self_method(image, psf, reg=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_psf_balance_kw_reg_replacer(args, kwargs, dispatchables):
    def self_method(image, psf, balance, reg=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            balance,
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


@create_skimage_restoration(_identity_replacer)
@all_of_type(ndarray)
@_get_docs
def ball_kernel(radius, ndim):
    return ()


@create_skimage_restoration(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def calibrate_denoiser(
    image,
    denoise_function,
    denoise_parameters,
    *,
    stride=4,
    approximate_loss=True,
    extra_output=False,
):
    return (image,)


@create_skimage_restoration(_x_replacer)
@all_of_type(ndarray)
@_get_docs
def cycle_spin(
    x,
    func,
    max_shifts,
    shift_steps=1,
    num_workers=None,
    multichannel=False,
    func_kw={},
    *,
    channel_axis=-1,
):
    return (x,)


@create_skimage_restoration(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def denoise_bilateral(
    image,
    win_size=None,
    sigma_color=None,
    sigma_spatial=1,
    bins=10000,
    mode='constant',
    cval=0,
    multichannel=False,
    *,
    channel_axis=None,
):
    return (image,)


@create_skimage_restoration(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def denoise_nl_means(
    image,
    patch_size=7,
    patch_distance=11,
    h=0.1,
    multichannel=False,
    fast_mode=True,
    sigma=0.0,
    *,
    preserve_range=False,
    channel_axis=None,
):
    return (image,)


@create_skimage_restoration(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def denoise_tv_bregman(
    image,
    weight=5.0,
    max_num_iter=100,
    eps=0.001,
    isotropic=True,
    *,
    channel_axis=None,
    multichannel=False,
):
    return (image,)


@create_skimage_restoration(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def denoise_tv_chambolle(
    image,
    weight=0.1,
    eps=0.0002,
    max_num_iter=200,
    multichannel=False,
    *,
    channel_axis=None,
):
    return (image,)


@create_skimage_restoration(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def denoise_wavelet(
    image,
    sigma=None,
    wavelet='db1',
    mode='soft',
    wavelet_levels=None,
    multichannel=False,
    convert2ycbcr=False,
    method='BayesShrink',
    rescale_sigma=True,
    *,
    channel_axis=None,
):
    return (image,)


@create_skimage_restoration(_shape_replacer)
@all_of_type(ndarray)
@_get_docs
def ellipsoid_kernel(shape, intensity):
    return (shape,)


@create_skimage_restoration(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def estimate_sigma(
    image, average_sigmas=False, multichannel=False, *, channel_axis=None
):
    return (image,)


@create_skimage_restoration(_image_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def inpaint_biharmonic(
    image,
    mask,
    multichannel=False,
    *,
    split_into_regions=False,
    channel_axis=None,
):
    return (image, mask)


@create_skimage_restoration(_image_psf_replacer)
@all_of_type(ndarray)
@_get_docs
def richardson_lucy(image, psf, num_iter=50, clip=True, filter_epsilon=None):
    return (image, psf)


@create_skimage_restoration(_image_kwonly_radius_kernel_replacer)
@all_of_type(ndarray)
@_get_docs
def rolling_ball(
    image, *, radius=100, kernel=None, nansafe=False, num_threads=None
):
    return (image, kernel)


@create_skimage_restoration(_image_psf_kw_reg_replacer)
@all_of_type(ndarray)
@_get_docs
def unsupervised_wiener(
    image,
    psf,
    reg=None,
    user_params=None,
    is_real=True,
    clip=True,
    *,
    random_state=None,
):
    return (image, psf, reg)


@create_skimage_restoration(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def unwrap_phase(image, wrap_around=False, seed=None):
    return (image,)


@create_skimage_restoration(_image_psf_balance_kw_reg_replacer)
@all_of_type(ndarray)
@_get_docs
def wiener(image, psf, balance, reg=None, is_real=True, clip=True):
    return (image, psf, reg)
