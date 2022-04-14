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

from skimage._backend import (_mark_output, _mark_scalar_or_array,
                              _mark_transform)

from skimage.transform import _api



__all__ = [
    # 'AffineTransform',
    # 'EssentialMatrixTransform',
    # 'EuclideanTransform',
    # 'FundamentalMatrixTransform',
    # 'PiecewiseAffineTransform',
    # 'PolynomialTransform',
    # 'ProjectiveTransform',
    # 'SimilarityTransform',
    'downscale_local_mean',
    'estimate_transform',
    'frt2',
    'hough_circle',
    'hough_circle_peaks',
    'hough_ellipse',
    'hough_line',
    'hough_line_peaks',
    'ifrt2',
    'integral_image',
    'integrate',
    'iradon',
    'iradon_sart',
    'matrix_transform',
    'order_angles_golden_ratio',
    'probabilistic_hough_line',
    'pyramid_expand',
    'pyramid_gaussian',
    'pyramid_laplacian',
    'pyramid_reduce',
    'radon',
    'rescale',
    'resize',
    'resize_local_mean',
    'rotate',
    'swirl',
    'warp',
    'warp_coords',
    'warp_polar',
]


create_skimage_transform = functools.partial(
    create_multimethod, domain="numpy.skimage.transform"
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


def _image_factors_replacer(args, kwargs, dispatchables):
    def self_method(image, factors, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _ttype_src_dst_replacer(args, kwargs, dispatchables):
    def self_method(ttype, src, dst, *args, **kwargs):
        kw_out = kwargs
        return (ttype, src, dst) + args, kw_out

    return self_method(*args, **kwargs)


def _a_replacer(args, kwargs, dispatchables):
    def self_method(a, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _hspaces_radii_replacer(args, kwargs, dispatchables):
    def self_method(hspaces, radii, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_theta_replacer(args, kwargs, dispatchables):
    def self_method(image, theta=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _hspace_angles_dists_replacer(args, kwargs, dispatchables):
    def self_method(hspace, angles, dists, *args, **kwargs):
        kw_out = kwargs
        return (
            dispatchables[0],
            dispatchables[1],
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kwonly_dtype_replacer(args, kwargs, dispatchables):
    def self_method(image, **kwargs):
        kw_out = kwargs.copy()
        if 'dtype' in kw_out:
            kw_out['dtype'] = dispatchables[1]
        return (dispatchables[0],), kw_out

    return self_method(*args, **kwargs)


def _ii_replacer(args, kwargs, dispatchables):
    def self_method(ii, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _radonimage_kw_theta_replacer(args, kwargs, dispatchables):
    def self_method(radon_image, theta=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _radonimage_kw_theta_image_projectionshifts_clip_relaxation_dtype_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        radon_image,
        theta=None,
        image=None,
        projection_shifts=None,
        clip=None,
        relaxation=0.15,
        dtype=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            dispatchables[2],
            dispatchables[3],
            clip,
            relaxation,
            dispatchables[4],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _coords_matrix_replacer(args, kwargs, dispatchables):
    def self_method(coords, matrix, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _theta_replacer(args, kwargs, dispatchables):
    def self_method(theta, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_threshold_linelength_linegap_theta_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        image,
        threshold=10,
        line_length=50,
        line_gap=10,
        theta=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            threshold,
            line_length,
            line_gap,
            dispatchables[1],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_outputshape_replacer(args, kwargs, dispatchables):
    def self_method(image, output_shape, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_angle_kw_resize_center_replacer(args, kwargs, dispatchables):
    def self_method(image, angle, resize=False, center=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            angle,
            resize,
            dispatchables[1],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_center_strength_radius_rotation_outputshape_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        image,
        center=None,
        strength=1,
        radius=100,
        rotation=0,
        output_shape=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            strength,
            radius,
            rotation,
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_inversemap_kw_mapargs_outputshape_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        image, inverse_map, *args, **kwargs
    ):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _coordmap_shape_kw_dtype_replacer(args, kwargs, dispatchables):
    def self_method(coord_map, shape, dtype='float64', *args, **kwargs):
        kw_out = kwargs.copy()
        return (coord_map, shape, dispatchables[0]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_center_kwonly_radius_outputshape_scaling_channelaxis_replacer(
    args, kwargs, dispatchables
):
    def self_method(image, center=None, **kwargs):
        kw_out = kwargs.copy()
        if 'output_shape' in kw_out:
            kw_out['output_shape'] = dispatchables[2]
        return (dispatchables[0], dispatchables[1]), kw_out

    return self_method(*args, **kwargs)


@create_skimage_transform(_image_factors_replacer)
@all_of_type(ndarray)
@_get_docs
def downscale_local_mean(image, factors, cval=0, clip=True):
    return (image, factors)


@create_skimage_transform(_ttype_src_dst_replacer)
@all_of_type(ndarray)
@_get_docs
def estimate_transform(ttype, src, dst, *args, **kwargs):
    return (kwargs,)


@create_skimage_transform(_a_replacer)
@all_of_type(ndarray)
@_get_docs
def frt2(a):
    return (a,)


@create_skimage_transform(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def hough_circle(image, radius, normalize=True, full_output=False):
    return (image,)


@create_skimage_transform(_hspaces_radii_replacer)
@all_of_type(ndarray)
@_get_docs
def hough_circle_peaks(
    hspaces,
    radii,
    min_xdistance=1,
    min_ydistance=1,
    threshold=None,
    num_peaks=np.inf,
    total_num_peaks=np.inf,
    normalize=False,
):
    return (hspaces, radii)


@create_skimage_transform(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def hough_ellipse(image, threshold=4, accuracy=1, min_size=4, max_size=None):
    return (image,)


@create_skimage_transform(_image_kw_theta_replacer)
@all_of_type(ndarray)
@_get_docs
def hough_line(image, theta=None):
    return (image, theta)


@create_skimage_transform(_hspace_angles_dists_replacer)
@all_of_type(ndarray)
@_get_docs
def hough_line_peaks(
    hspace,
    angles,
    dists,
    min_distance=9,
    min_angle=10,
    threshold=None,
    num_peaks=np.inf,
):
    return (hspace, angles, dists)


@create_skimage_transform(_a_replacer)
@all_of_type(ndarray)
@_get_docs
def ifrt2(a):
    return (a,)


@create_skimage_transform(_image_kwonly_dtype_replacer)
@all_of_type(ndarray)
@_get_docs
def integral_image(image, *, dtype=None):
    return (image, Dispatchable(dtype, np.dtype))


@create_skimage_transform(_ii_replacer)
@all_of_type(ndarray)
@_get_docs
def integrate(ii, start, end):
    return (ii,)


@create_skimage_transform(_radonimage_kw_theta_replacer)
@all_of_type(ndarray)
@_get_docs
def iradon(
    radon_image,
    theta=None,
    output_size=None,
    filter_name='ramp',
    interpolation='linear',
    circle=True,
    preserve_range=True,
):
    return (radon_image, theta)


@create_skimage_transform(
    _radonimage_kw_theta_image_projectionshifts_clip_relaxation_dtype_replacer
)
@all_of_type(ndarray)
@_get_docs
def iradon_sart(
    radon_image,
    theta=None,
    image=None,
    projection_shifts=None,
    clip=None,
    relaxation=0.15,
    dtype=None,
):
    return (
        radon_image,
        theta,
        image,
        projection_shifts,
        Dispatchable(dtype, np.dtype),
    )


@create_skimage_transform(_coords_matrix_replacer)
@all_of_type(ndarray)
@_get_docs
def matrix_transform(coords, matrix):
    return (coords, matrix)


@create_skimage_transform(_theta_replacer)
@all_of_type(ndarray)
@_get_docs
def order_angles_golden_ratio(theta):
    return (theta,)


@create_skimage_transform(
    _image_kw_threshold_linelength_linegap_theta_replacer
)
@all_of_type(ndarray)
@_get_docs
def probabilistic_hough_line(
    image, threshold=10, line_length=50, line_gap=10, theta=None, seed=None
):
    return (image, theta)


@create_skimage_transform(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def pyramid_expand(
    image,
    upscale=2,
    sigma=None,
    order=1,
    mode='reflect',
    cval=0,
    preserve_range=False,
    *,
    channel_axis=None,
):
    return (image,)


@create_skimage_transform(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def pyramid_gaussian(
    image,
    max_layer=-1,
    downscale=2,
    sigma=None,
    order=1,
    mode='reflect',
    cval=0,
    preserve_range=False,
    *,
    channel_axis=None,
):
    return (image,)


@create_skimage_transform(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def pyramid_laplacian(
    image,
    max_layer=-1,
    downscale=2,
    sigma=None,
    order=1,
    mode='reflect',
    cval=0,
    preserve_range=False,
    *,
    channel_axis=None,
):
    return (image,)


@create_skimage_transform(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def pyramid_reduce(
    image,
    downscale=2,
    sigma=None,
    order=1,
    mode='reflect',
    cval=0,
    preserve_range=False,
    *,
    channel_axis=None,
):
    return (image,)


@create_skimage_transform(_image_kw_theta_replacer)
@all_of_type(ndarray)
@_get_docs
def radon(image, theta=None, circle=True, *, preserve_range=False):
    return (image, theta)


@create_skimage_transform(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def rescale(
    image,
    scale,
    order=None,
    mode='reflect',
    cval=0,
    clip=True,
    preserve_range=False,

    anti_aliasing=None,
    anti_aliasing_sigma=None,
    *,
    channel_axis=None,
):
    return (image,)


@create_skimage_transform(_image_outputshape_replacer)
@all_of_type(ndarray)
@_get_docs
def resize(
    image,
    output_shape,
    order=None,
    mode='reflect',
    cval=0,
    clip=True,
    preserve_range=False,
    anti_aliasing=None,
    anti_aliasing_sigma=None,
):
    return (image, output_shape)


@create_skimage_transform(_image_outputshape_replacer)
@all_of_type(ndarray)
@_get_docs
def resize_local_mean(
    image,
    output_shape,
    grid_mode=True,
    preserve_range=False,
    *,
    channel_axis=None,
):
    return (image, output_shape)


@create_skimage_transform(_image_angle_kw_resize_center_replacer)
@all_of_type(ndarray)
@_get_docs
def rotate(
    image,
    angle,
    resize=False,
    center=None,
    order=None,
    mode='constant',
    cval=0,
    clip=True,
    preserve_range=False,
):
    return (image, center)


@create_skimage_transform(
    _image_kw_center_strength_radius_rotation_outputshape_replacer
)
@all_of_type(ndarray)
@_get_docs
def swirl(
    image,
    center=None,
    strength=1,
    radius=100,
    rotation=0,
    output_shape=None,
    order=None,
    mode='reflect',
    cval=0,
    clip=True,
    preserve_range=False,
):
    return (image, center, output_shape)


@create_skimage_transform(_image_inversemap_kw_mapargs_outputshape_replacer)
@all_of_type(ndarray)
@_get_docs
def warp(
    image,
    inverse_map,
    map_args={},
    output_shape=None,
    order=None,
    mode='constant',
    cval=0.0,
    clip=True,
    preserve_range=False,
):
    return (image, _mark_transform(inverse_map))


@create_skimage_transform(_coordmap_shape_kw_dtype_replacer)
@all_of_type(ndarray)
@_get_docs
def warp_coords(coord_map, shape, dtype='float64'):
    return (Dispatchable(dtype, np.dtype),)


@create_skimage_transform(
    _image_kw_center_kwonly_radius_outputshape_scaling_channelaxis_replacer
)
@all_of_type(ndarray)
@_get_docs
def warp_polar(
    image,
    center=None,
    *,
    radius=None,
    output_shape=None,
    scaling='linear',
    channel_axis=None,
    **kwargs,
):
    return (image, center, output_shape, kwargs)
