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
    'active_contour',
    'chan_vese',
    'checkerboard_level_set',
    'clear_border',
    'disk_level_set',
    'expand_labels',
    'felzenszwalb',
    'find_boundaries',
    'flood',
    'flood_fill',
    'inverse_gaussian_gradient',
    'join_segmentations',
    'mark_boundaries',
    'morphological_chan_vese',
    'morphological_geodesic_active_contour',
    'quickshift',
    'random_walker',
    'relabel_sequential',
    'slic',
    'watershed',
]


create_skimage_segmentation = functools.partial(
    create_multimethod, domain="numpy.skimage.segmentation"
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


def _image_snake_replacer(args, kwargs, dispatchables):
    def self_method(image, snake, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_mu_lambda1_lambda2_tol_maxnumiter_dt_initlevelset_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        image,
        mu=0.25,
        lambda1=1.0,
        lambda2=1.0,
        tol=0.001,
        max_num_iter=500,
        dt=0.5,
        init_level_set='checkerboard',
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            mu,
            lambda1,
            lambda2,
            tol,
            max_num_iter,
            dt,
            dispatchables[1],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _labels_kw_buffersize_bgval_inplace_mask_kwonly_out_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        labels, buffer_size=0, bgval=0, in_place=False, mask=None, **kwargs
    ):
        kw_out = kwargs.copy()
        if 'out' in kw_out:
            kw_out['out'] = dispatchables[2]
        return (
            dispatchables[0],
            buffer_size,
            bgval,
            in_place,
            dispatchables[1],
        ), kw_out

    return self_method(*args, **kwargs)


def _labelimage_replacer(args, kwargs, dispatchables):
    def self_method(label_image, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_scale_sigma_replacer(args, kwargs, dispatchables):
    def self_method(image, scale=1, sigma=0.8, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], scale, dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _labelimg_kw_connectivity_replacer(args, kwargs, dispatchables):
    def self_method(label_img, connectivity=1, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_alpha_sigma_replacer(args, kwargs, dispatchables):
    def self_method(image, alpha=100.0, sigma=5.0, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], alpha, dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_labelimg_kw_color_replacer(args, kwargs, dispatchables):
    def self_method(image, label_img, color=(1, 1, 0), *args, **kwargs):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            color,
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_numiter_kw_initlevelset_replacer(args, kwargs, dispatchables):
    def self_method(
        image, num_iter, init_level_set='checkerboard', *args, **kwargs
    ):
        kw_out = kwargs.copy()
        return (dispatchables[0], num_iter, dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _gimage_numiter_kw_initlevelset_replacer(args, kwargs, dispatchables):
    def self_method(gimage, num_iter, init_level_set='disk', *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], num_iter, dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_ratio_kernelsize_maxdist_returntree_sigma_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        image,
        ratio=1.0,
        kernel_size=5,
        max_dist=10,
        return_tree=False,
        sigma=0,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            ratio,
            kernel_size,
            max_dist,
            return_tree,
            dispatchables[1],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _data_labels_kw_beta_mode_tol_copy_returnfullprob_spacing_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        data,
        labels,
        beta=130,
        mode='cg_j',
        tol=0.001,
        copy=True,
        return_full_prob=False,
        spacing=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            beta,
            mode,
            tol,
            copy,
            return_full_prob,
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _labelfield_kw_offset_replacer(args, kwargs, dispatchables):
    def self_method(label_field, offset=1, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_nsegments_compactness_maxnumiter_sigma_spacing_convert2lab_enforceconnectivity_minsizefactor_maxsizefactor_sliczero_startlabel_mask_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        image,
        n_segments=100,
        compactness=10.0,
        max_num_iter=10,
        sigma=0,
        spacing=None,
        convert2lab=None,
        enforce_connectivity=True,
        min_size_factor=0.5,
        max_size_factor=3,
        slic_zero=False,
        start_label=1,
        mask=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            n_segments,
            compactness,
            max_num_iter,
            dispatchables[1],
            dispatchables[2],
            convert2lab,
            enforce_connectivity,
            min_size_factor,
            max_size_factor,
            slic_zero,
            start_label,
            dispatchables[3],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_markers_connectivity_offset_mask_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        image,
        markers=None,
        connectivity=1,
        offset=None,
        mask=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            dispatchables[2],
            dispatchables[3],
            dispatchables[4],
        ) + args, kw_out

    return self_method(*args, **kwargs)


@create_skimage_segmentation(_image_snake_replacer)
@all_of_type(ndarray)
@_get_docs
def active_contour(
    image,
    snake,
    alpha=0.01,
    beta=0.1,
    w_line=0,
    w_edge=1,
    gamma=0.01,
    max_px_move=1.0,
    max_num_iter=2500,
    convergence=0.1,
    *,
    boundary_condition='periodic',
    coordinates='rc',
):
    return (image, snake)


@create_skimage_segmentation(
    _image_kw_mu_lambda1_lambda2_tol_maxnumiter_dt_initlevelset_replacer
)
@all_of_type(ndarray)
@_get_docs
def chan_vese(
    image,
    mu=0.25,
    lambda1=1.0,
    lambda2=1.0,
    tol=0.001,
    max_num_iter=500,
    dt=0.5,
    init_level_set='checkerboard',
    extended_output=False,
):
    return (image, init_level_set)


@create_skimage_segmentation(_identity_replacer)
@all_of_type(ndarray)
@_get_docs
def checkerboard_level_set(image_shape, square_size=5):
    return ()


@create_skimage_segmentation(
    _labels_kw_buffersize_bgval_inplace_mask_kwonly_out_replacer
)
@all_of_type(ndarray)
@_get_docs
def clear_border(
    labels, buffer_size=0, bgval=0, in_place=False, mask=None, *, out=None
):
    return (labels, mask, _mark_output(out))


@create_skimage_segmentation(_identity_replacer)
@all_of_type(ndarray)
@_get_docs
def disk_level_set(image_shape, *, center=None, radius=None):
    return ()


@create_skimage_segmentation(_labelimage_replacer)
@all_of_type(ndarray)
@_get_docs
def expand_labels(label_image, distance=1):
    return (label_image,)


@create_skimage_segmentation(_image_kw_scale_sigma_replacer)
@all_of_type(ndarray)
@_get_docs
def felzenszwalb(
    image,
    scale=1,
    sigma=0.8,
    min_size=20,
    *,
    channel_axis=-1,
):
    return (image, sigma)


@create_skimage_segmentation(_labelimg_kw_connectivity_replacer)
@all_of_type(ndarray)
@_get_docs
def find_boundaries(label_img, connectivity=1, mode='thick', background=0):
    return (label_img, connectivity)


@create_skimage_segmentation(_image_kw_alpha_sigma_replacer)
@all_of_type(ndarray)
@_get_docs
def inverse_gaussian_gradient(image, alpha=100.0, sigma=5.0):
    return (image, sigma)


@create_skimage_segmentation(_identity_replacer)
@all_of_type(ndarray)
@_get_docs
def join_segmentations(s1, s2):
    return ()


@create_skimage_segmentation(_image_labelimg_kw_color_replacer)
@all_of_type(ndarray)
@_get_docs
def mark_boundaries(
    image,
    label_img,
    color=(1, 1, 0),
    outline_color=None,
    mode='outer',
    background_label=0,
):
    return (image, label_img)


@create_skimage_segmentation(_image_numiter_kw_initlevelset_replacer)
@all_of_type(ndarray)
@_get_docs
def morphological_chan_vese(
    image,
    num_iter,
    init_level_set='checkerboard',
    smoothing=1,
    lambda1=1,
    lambda2=1,
    iter_callback=lambda x: None,
):
    return (image, init_level_set)


@create_skimage_segmentation(_gimage_numiter_kw_initlevelset_replacer)
@all_of_type(ndarray)
@_get_docs
def morphological_geodesic_active_contour(
    gimage,
    num_iter,
    init_level_set='disk',
    smoothing=1,
    threshold='auto',
    balloon=0,
    iter_callback=lambda x: None,
):
    return (gimage, init_level_set)


@create_skimage_segmentation(
    _image_kw_ratio_kernelsize_maxdist_returntree_sigma_replacer
)
@all_of_type(ndarray)
@_get_docs
def quickshift(
    image,
    ratio=1.0,
    kernel_size=5,
    max_dist=10,
    return_tree=False,
    sigma=0,
    convert2lab=True,
    random_seed=42,
    *,
    channel_axis=-1,
):
    return (image, sigma)


@create_skimage_segmentation(
    _data_labels_kw_beta_mode_tol_copy_returnfullprob_spacing_replacer
)
@all_of_type(ndarray)
@_get_docs
def random_walker(
    data,
    labels,
    beta=130,
    mode='cg_j',
    tol=0.001,
    copy=True,
    return_full_prob=False,
    spacing=None,
    *,
    prob_tol=0.001,
    channel_axis=None,
):
    return (data, labels, spacing)


@create_skimage_segmentation(_labelfield_kw_offset_replacer)
@all_of_type(ndarray)
@_get_docs
def relabel_sequential(label_field, offset=1):
    return (label_field, offset)


@create_skimage_segmentation(
    _image_kw_nsegments_compactness_maxnumiter_sigma_spacing_convert2lab_enforceconnectivity_minsizefactor_maxsizefactor_sliczero_startlabel_mask_replacer
)
@all_of_type(ndarray)
@_get_docs
def slic(
    image,
    n_segments=100,
    compactness=10.0,
    max_num_iter=10,
    sigma=0,
    spacing=None,
    convert2lab=None,
    enforce_connectivity=True,
    min_size_factor=0.5,
    max_size_factor=3,
    slic_zero=False,
    start_label=1,
    mask=None,
    *,
    channel_axis=-1,
):
    return (image, sigma, spacing, mask)


@create_skimage_segmentation(
    _image_kw_markers_connectivity_offset_mask_replacer
)
@all_of_type(ndarray)
@_get_docs
def watershed(
    image,
    markers=None,
    connectivity=1,
    offset=None,
    mask=None,
    compactness=0,
    watershed_line=False,
):
    return (image, markers, connectivity, offset, mask)
