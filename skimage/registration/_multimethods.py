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

__all__ = ['optical_flow_ilk', 'optical_flow_tvl1', 'phase_cross_correlation']


create_skimage_registration = functools.partial(
    create_multimethod, domain="numpy.skimage.registration"
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


def _referenceimage_movingimage_kwonly_radius_numwarp_gaussian_prefilter_dtype_replacer(
    args, kwargs, dispatchables
):
    def self_method(reference_image, moving_image, **kwargs):
        kw_out = kwargs.copy()
        if 'dtype' in kw_out:
            kw_out['dtype'] = dispatchables[2]
        return (dispatchables[0], dispatchables[1]), kw_out

    return self_method(*args, **kwargs)


def _referenceimage_movingimage_kwonly_attachment_tightness_numwarp_numiter_tol_prefilter_dtype_replacer(
    args, kwargs, dispatchables
):
    def self_method(reference_image, moving_image, **kwargs):
        kw_out = kwargs.copy()
        if 'dtype' in kw_out:
            kw_out['dtype'] = dispatchables[2]
        return (dispatchables[0], dispatchables[1]), kw_out

    return self_method(*args, **kwargs)


def _referenceimage_movingimage_replacer(args, kwargs, dispatchables):
    def self_method(reference_image, moving_image, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


@create_skimage_registration(
    _referenceimage_movingimage_kwonly_radius_numwarp_gaussian_prefilter_dtype_replacer
)
@all_of_type(ndarray)
@_get_docs
def optical_flow_ilk(
    reference_image,
    moving_image,
    *,
    radius=7,
    num_warp=10,
    gaussian=False,
    prefilter=False,
    dtype='float32',
):
    return (reference_image, moving_image, Dispatchable(dtype, np.dtype))


@create_skimage_registration(
    _referenceimage_movingimage_kwonly_attachment_tightness_numwarp_numiter_tol_prefilter_dtype_replacer
)
@all_of_type(ndarray)
@_get_docs
def optical_flow_tvl1(
    reference_image,
    moving_image,
    *,
    attachment=15,
    tightness=0.3,
    num_warp=5,
    num_iter=10,
    tol=0.0001,
    prefilter=False,
    dtype='float32',
):
    return (reference_image, moving_image, Dispatchable(dtype, np.dtype))


@create_skimage_registration(_referenceimage_movingimage_replacer)
@all_of_type(ndarray)
@_get_docs
def phase_cross_correlation(
    reference_image,
    moving_image,
    *,
    upsample_factor=1,
    space='real',
    return_error=True,
    reference_mask=None,
    moving_mask=None,
    overlap_ratio=0.3,
    normalization='phase',
):
    return (reference_image, moving_image)
