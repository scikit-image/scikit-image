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
    'CircleModel',
    'EllipseModel',
    'LineModelND',
    'approximate_polygon',
    'block_reduce',
    'blur_effect',
    'euler_number',
    'find_contours',
    'grid_points_in_poly',
    'inertia_tensor',
    'inertia_tensor_eigvals',
    'label',
    'marching_cubes',
    'mesh_surface_area',
    'moments',
    'moments_central',
    'moments_coords',
    'moments_coords_central',
    'moments_hu',
    'moments_normalized',
    'perimeter',
    'perimeter_crofton',
    'points_in_poly',
    'profile_line',
    'ransac',
    'regionprops',
    'regionprops_table',
    'shannon_entropy',
    'subdivide_polygon',
]


create_skimage_measure = functools.partial(
    create_multimethod, domain="numpy.skimage.measure"
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


def _coords_replacer(args, kwargs, dispatchables):
    def self_method(coords, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_blocksize_replacer(args, kwargs, dispatchables):
    def self_method(image, block_size=2, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_level_fullyconnected_positiveorientation_kwonly_mask_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        image,
        level=None,
        fully_connected='low',
        positive_orientation='low',
        **kwargs,
    ):
        kw_out = kwargs.copy()
        if 'mask' in kw_out:
            kw_out['mask'] = dispatchables[1]
        return (
            dispatchables[0],
            level,
            fully_connected,
            positive_orientation,
        ), kw_out

    return self_method(*args, **kwargs)


def _shape_verts_replacer(args, kwargs, dispatchables):
    def self_method(shape, verts, *args, **kwargs):
        kw_out = kwargs
        return (shape, dispatchables[0]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_mu_replacer(args, kwargs, dispatchables):
    def self_method(image, mu=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_mu_T_replacer(args, kwargs, dispatchables):
    def self_method(image, mu=None, T=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _labelimage_replacer(args, kwargs, dispatchables):
    def self_method(label_image, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _volume_kw_level_kwonly_spacing_gradientdirection_stepsize_allowdegenerate_method_mask_replacer(
    args, kwargs, dispatchables
):
    def self_method(volume, level=None, **kwargs):
        kw_out = kwargs.copy()
        if 'mask' in kw_out:
            kw_out['mask'] = dispatchables[1]
        return (dispatchables[0], level), kw_out

    return self_method(*args, **kwargs)


def _verts_faces_replacer(args, kwargs, dispatchables):
    def self_method(verts, faces, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _nu_replacer(args, kwargs, dispatchables):
    def self_method(nu, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _mu_replacer(args, kwargs, dispatchables):
    def self_method(mu, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _points_verts_replacer(args, kwargs, dispatchables):
    def self_method(points, verts, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_src_dst_replacer(args, kwargs, dispatchables):
    def self_method(image, src, dst, *args, **kwargs):
        kw_out = kwargs
        return (
            dispatchables[0],
            dispatchables[1],
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _data_modelclass_minsamples_residualthreshold_kw_isdatavalid_ismodelvalid_maxtrials_stopsamplenum_stopresidualssum_stopprobability_randomstate_initialinliers_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        data,
        model_class,
        min_samples,
        residual_threshold,
        is_data_valid=None,
        is_model_valid=None,
        max_trials=100,
        stop_sample_num=np.inf,
        stop_residuals_sum=0,
        stop_probability=1,
        random_state=None,
        initial_inliers=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            model_class,
            min_samples,
            residual_threshold,
            is_data_valid,
            is_model_valid,
            max_trials,
            stop_sample_num,
            stop_residuals_sum,
            stop_probability,
            random_state,
            dispatchables[1],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _labelimage_kw_intensityimage_replacer(args, kwargs, dispatchables):
    def self_method(label_image, intensity_image=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


@create_skimage_measure(_coords_replacer)
@all_of_type(ndarray)
@_get_docs
def approximate_polygon(coords, tolerance):
    return (coords,)


@create_skimage_measure(_image_kw_blocksize_replacer)
@all_of_type(ndarray)
@_get_docs
def block_reduce(image, block_size=2, func=sum, cval=0, func_kwargs=None):
    return (image, block_size)


@create_skimage_measure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def blur_effect(image, h_size=11, channel_axis=None, reduce_func=np.max):
    return (image,)


@create_skimage_measure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def centroid(image, *, spacing=None):
    return (image,)


@create_skimage_measure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def euler_number(image, connectivity=None):
    return (image,)


@create_skimage_measure(
    _image_kw_level_fullyconnected_positiveorientation_kwonly_mask_replacer
)
@all_of_type(ndarray)
@_get_docs
def find_contours(
    image,
    level=None,
    fully_connected='low',
    positive_orientation='low',
    *,
    mask=None,
):
    return (image, mask)


@create_skimage_measure(_shape_verts_replacer)
@all_of_type(ndarray)
@_get_docs
def grid_points_in_poly(shape, verts):
    return (verts,)


@create_skimage_measure(_image_kw_mu_replacer)
@all_of_type(ndarray)
@_get_docs
def inertia_tensor(image, mu=None, *, spacing=None):
    return (image, mu)


@create_skimage_measure(_image_kw_mu_T_replacer)
@all_of_type(ndarray)
@_get_docs
def inertia_tensor_eigvals(image, mu=None, T=None, *, spacing=None):
    return (image, mu, T)


@create_skimage_measure(_labelimage_replacer)
@all_of_type(ndarray)
@_get_docs
def label(label_image, background=None, return_num=False, connectivity=None):
    return (label_image,)


@create_skimage_measure(
    _volume_kw_level_kwonly_spacing_gradientdirection_stepsize_allowdegenerate_method_mask_replacer
)
@all_of_type(ndarray)
@_get_docs
def marching_cubes(
    volume,
    level=None,
    *,
    spacing=(1.0, 1.0, 1.0),
    gradient_direction='descent',
    step_size=1,
    allow_degenerate=True,
    method='lewiner',
    mask=None,
):
    return (volume, mask)


@create_skimage_measure(_verts_faces_replacer)
@all_of_type(ndarray)
@_get_docs
def mesh_surface_area(verts, faces):
    return (verts, faces)


@create_skimage_measure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def moments(image, order=3, *, spacing=None):
    return (image,)


@create_skimage_measure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def moments_central(image, center=None, order=3, *, spacing=None, **kwargs):
    return (image,)


@create_skimage_measure(_coords_replacer)
@all_of_type(ndarray)
@_get_docs
def moments_coords(coords, order=3):
    return (coords,)


@create_skimage_measure(_coords_replacer)
@all_of_type(ndarray)
@_get_docs
def moments_coords_central(coords, center=None, order=3):
    return (coords,)


@create_skimage_measure(_nu_replacer)
@all_of_type(ndarray)
@_get_docs
def moments_hu(nu):
    return (nu,)


@create_skimage_measure(_mu_replacer)
@all_of_type(ndarray)
@_get_docs
def moments_normalized(mu, order=3, *, spacing=None):
    return (mu,)


@create_skimage_measure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def perimeter(image, neighbourhood=4):
    return (image,)


@create_skimage_measure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def perimeter_crofton(image, directions=4):
    return (image,)


@create_skimage_measure(_points_verts_replacer)
@all_of_type(ndarray)
@_get_docs
def points_in_poly(points, verts):
    return (points, verts)


@create_skimage_measure(_image_src_dst_replacer)
@all_of_type(ndarray)
@_get_docs
def profile_line(
    image,
    src,
    dst,
    linewidth=1,
    order=None,
    mode='reflect',
    cval=0.0,
    *,
    reduce_func=np.mean,
):
    return (image, src, dst)


@create_skimage_measure(
    _data_modelclass_minsamples_residualthreshold_kw_isdatavalid_ismodelvalid_maxtrials_stopsamplenum_stopresidualssum_stopprobability_randomstate_initialinliers_replacer
)
@all_of_type(ndarray)
@_get_docs
def ransac(
    data,
    model_class,
    min_samples,
    residual_threshold,
    is_data_valid=None,
    is_model_valid=None,
    max_trials=100,
    stop_sample_num=np.inf,
    stop_residuals_sum=0,
    stop_probability=1,
    random_state=None,
    initial_inliers=None,
):
    return (data, initial_inliers)


@create_skimage_measure(_labelimage_kw_intensityimage_replacer)
@all_of_type(ndarray)
@_get_docs
def regionprops(
    label_image,
    intensity_image=None,
    cache=True,
    coordinates=None,
    *,
    extra_properties=None,
    spacing=None,
):
    return (label_image, intensity_image)


@create_skimage_measure(_labelimage_kw_intensityimage_replacer)
@all_of_type(ndarray)
@_get_docs
def regionprops_table(
    label_image,
    intensity_image=None,
    properties=('label', 'bbox'),
    *,
    cache=True,
    separator='-',
    extra_properties=None,
    spacing=None,
):
    return (label_image, intensity_image)


@create_skimage_measure(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def shannon_entropy(image, base=2):
    return (image,)


@create_skimage_measure(_coords_replacer)
@all_of_type(ndarray)
@_get_docs
def subdivide_polygon(coords, degree=2, preserve_ends=False):
    return (coords,)
