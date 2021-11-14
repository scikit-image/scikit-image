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
    'BRIEF',
    'CENSURE',
    'Cascade',
    'ORB',
    'blob_dog',
    'blob_doh',
    'blob_log',
    'canny',
    'corner_fast',
    'corner_foerstner',
    'corner_harris',
    'corner_kitchen_rosenfeld',
    'corner_moravec',
    'corner_orientations',
    'corner_peaks',
    'corner_shi_tomasi',
    'corner_subpix',
    'daisy',
    'draw_haar_like_feature',
    'draw_multiblock_lbp',
    'graycomatrix',
    'graycoprops',
    'greycomatrix',
    'greycoprops',
    'haar_like_feature',
    'haar_like_feature_coord',
    'hessian_matrix',
    'hessian_matrix_det',
    'hessian_matrix_eigvals',
    'hog',
    'local_binary_pattern',
    'match_descriptors',
    'match_template',
    'multiblock_lbp',
    'multiscale_basic_features',
    'peak_local_max',
    'plot_matches',
    'shape_index',
    'structure_tensor',
    'structure_tensor_eigenvalues',
    'structure_tensor_eigvals',
]


create_skimage_feature = functools.partial(
    create_multimethod, domain="numpy.skimage.feature"
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


def _image_kw_sigma_lowthreshold_highthreshold_mask_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        image,
        sigma=1.0,
        low_threshold=None,
        high_threshold=None,
        mask=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            low_threshold,
            high_threshold,
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_sigma_replacer(args, kwargs, dispatchables):
    def self_method(image, sigma=1, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_method_k_eps_sigma_replacer(args, kwargs, dispatchables):
    def self_method(
        image, method='k', k=0.05, eps=1e-06, sigma=1, *args, **kwargs
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            method,
            k,
            eps,
            dispatchables[1],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_corners_mask_replacer(args, kwargs, dispatchables):
    def self_method(image, corners, mask, *args, **kwargs):
        kw_out = kwargs
        return (
            dispatchables[0],
            dispatchables[1],
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_mindistance_thresholdabs_thresholdrel_excludeborder_indices_numpeaks_footprint_labels_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        image,
        min_distance=1,
        threshold_abs=None,
        threshold_rel=None,
        exclude_border=True,
        indices=True,
        num_peaks=np.inf,
        footprint=None,
        labels=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            min_distance,
            threshold_abs,
            threshold_rel,
            exclude_border,
            indices,
            num_peaks,
            dispatchables[1],
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_corners_replacer(args, kwargs, dispatchables):
    def self_method(image, corners, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_step_radius_rings_histograms_orientations_normalization_sigmas_ringradii_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        image,
        step=4,
        radius=15,
        rings=3,
        histograms=8,
        orientations=8,
        normalization='l1',
        sigmas=None,
        ring_radii=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            step,
            radius,
            rings,
            histograms,
            orientations,
            normalization,
            dispatchables[1],
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_r_c_width_height_featurecoord_replacer(args, kwargs, dispatchables):
    def self_method(
        image, r, c, width, height, feature_coord, *args, **kwargs
    ):
        kw_out = kwargs
        return (
            dispatchables[0],
            r,
            c,
            width,
            height,
            dispatchables[1],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_distances_angles_replacer(args, kwargs, dispatchables):
    def self_method(image, distances, angles, *args, **kwargs):
        kw_out = kwargs
        return (
            dispatchables[0],
            dispatchables[1],
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _P_replacer(args, kwargs, dispatchables):
    def self_method(P, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _intimage_r_c_width_height_kw_featuretype_featurecoord_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        int_image,
        r,
        c,
        width,
        height,
        feature_type=None,
        feature_coord=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            r,
            c,
            width,
            height,
            feature_type,
            dispatchables[1],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _Helems_replacer(args, kwargs, dispatchables):
    def self_method(H_elems, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _image_P_replacer(args, kwargs, dispatchables):
    def self_method(image, P, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _descriptors1_descriptors2_replacer(args, kwargs, dispatchables):
    def self_method(descriptors1, descriptors2, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_template_replacer(args, kwargs, dispatchables):
    def self_method(image, template, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _intimage_replacer(args, kwargs, dispatchables):
    def self_method(int_image, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _ax_image1_image2_keypoints1_keypoints2_matches_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        ax, image1, image2, keypoints1, keypoints2, matches, *args, **kwargs
    ):
        kw_out = kwargs
        return (
            ax,
            dispatchables[0],
            dispatchables[1],
            dispatchables[2],
            dispatchables[3],
            dispatchables[4],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _Aelems_replacer(args, kwargs, dispatchables):
    def self_method(A_elems, *args, **kwargs):
        kw_out = kwargs
        return (dispatchables[0],) + args, kw_out

    return self_method(*args, **kwargs)


def _Axx_Axy_Ayy_replacer(args, kwargs, dispatchables):
    def self_method(Axx, Axy, Ayy, *args, **kwargs):
        kw_out = kwargs
        return (
            dispatchables[0],
            dispatchables[1],
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


@create_skimage_feature(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def blob_dog(
    image,
    min_sigma=1,
    max_sigma=50,
    sigma_ratio=1.6,
    threshold=0.5,
    overlap=0.5,
    *,
    threshold_rel=None,
    exclude_border=False,
):
    return (image,)


@create_skimage_feature(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def blob_doh(
    image,
    min_sigma=1,
    max_sigma=30,
    num_sigma=10,
    threshold=0.01,
    overlap=0.5,
    log_scale=False,
    *,
    threshold_rel=None,
):
    return (image,)


@create_skimage_feature(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def blob_log(
    image,
    min_sigma=1,
    max_sigma=50,
    num_sigma=10,
    threshold=0.2,
    overlap=0.5,
    log_scale=False,
    *,
    threshold_rel=None,
    exclude_border=False,
):
    return (image,)


@create_skimage_feature(
    _image_kw_sigma_lowthreshold_highthreshold_mask_replacer
)
@all_of_type(ndarray)
@_get_docs
def canny(
    image,
    sigma=1.0,
    low_threshold=None,
    high_threshold=None,
    mask=None,
    use_quantiles=False,
    *,
    mode='constant',
    cval=0.0,
):
    return (image, sigma, mask)


@create_skimage_feature(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def corner_fast(image, n=12, threshold=0.15):
    return (image,)


@create_skimage_feature(_image_kw_sigma_replacer)
@all_of_type(ndarray)
@_get_docs
def corner_foerstner(image, sigma=1):
    return (image, sigma)


@create_skimage_feature(_image_kw_method_k_eps_sigma_replacer)
@all_of_type(ndarray)
@_get_docs
def corner_harris(image, method='k', k=0.05, eps=1e-06, sigma=1):
    return (image, sigma)


@create_skimage_feature(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def corner_kitchen_rosenfeld(image, mode='constant', cval=0):
    return (image,)


@create_skimage_feature(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def corner_moravec(image, window_size=1):
    return (image,)


@create_skimage_feature(_image_corners_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def corner_orientations(image, corners, mask):
    return (image, corners, mask)


@create_skimage_feature(
    _image_kw_mindistance_thresholdabs_thresholdrel_excludeborder_indices_numpeaks_footprint_labels_replacer
)
@all_of_type(ndarray)
@_get_docs
def corner_peaks(
    image,
    min_distance=1,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=True,
    indices=True,
    num_peaks=np.inf,
    footprint=None,
    labels=None,
    *,
    num_peaks_per_label=np.inf,
    p_norm=np.inf,
):
    return (image, footprint, labels)


@create_skimage_feature(_image_kw_sigma_replacer)
@all_of_type(ndarray)
@_get_docs
def corner_shi_tomasi(image, sigma=1):
    return (image, sigma)


@create_skimage_feature(_image_corners_replacer)
@all_of_type(ndarray)
@_get_docs
def corner_subpix(image, corners, window_size=11, alpha=0.99):
    return (image, corners)


@create_skimage_feature(
    _image_kw_step_radius_rings_histograms_orientations_normalization_sigmas_ringradii_replacer
)
@all_of_type(ndarray)
@_get_docs
def daisy(
    image,
    step=4,
    radius=15,
    rings=3,
    histograms=8,
    orientations=8,
    normalization='l1',
    sigmas=None,
    ring_radii=None,
    visualize=False,
):
    return (image, sigmas, ring_radii)


@create_skimage_feature(_image_r_c_width_height_featurecoord_replacer)
@all_of_type(ndarray)
@_get_docs
def draw_haar_like_feature(
    image,
    r,
    c,
    width,
    height,
    feature_coord,
    color_positive_block=(1.0, 0.0, 0.0),
    color_negative_block=(0.0, 1.0, 0.0),
    alpha=0.5,
    max_n_features=None,
    random_state=None,
):
    return (image, feature_coord)


@create_skimage_feature(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def draw_multiblock_lbp(
    image,
    r,
    c,
    width,
    height,
    lbp_code=0,
    color_greater_block=(1, 1, 1),
    color_less_block=(0, 0.69, 0.96),
    alpha=0.5,
):
    return (image,)


@create_skimage_feature(_image_distances_angles_replacer)
@all_of_type(ndarray)
@_get_docs
def graycomatrix(
    image, distances, angles, levels=None, symmetric=False, normed=False
):
    return (image, distances, angles)


@create_skimage_feature(_P_replacer)
@all_of_type(ndarray)
@_get_docs
def graycoprops(P, prop='contrast'):
    return (P,)


@create_skimage_feature(_image_distances_angles_replacer)
@all_of_type(ndarray)
@_get_docs
def greycomatrix(
    image, distances, angles, levels=None, symmetric=False, normed=False
):
    return (image, distances, angles)


@create_skimage_feature(_P_replacer)
@all_of_type(ndarray)
@_get_docs
def greycoprops(P, prop='contrast'):
    return (P,)


@create_skimage_feature(
    _intimage_r_c_width_height_kw_featuretype_featurecoord_replacer
)
@all_of_type(ndarray)
@_get_docs
def haar_like_feature(
    int_image, r, c, width, height, feature_type=None, feature_coord=None
):
    return (int_image, feature_coord)


@create_skimage_feature(_identity_replacer)
@all_of_type(ndarray)
@_get_docs
def haar_like_feature_coord(width, height, feature_type=None):
    return ()


@create_skimage_feature(_image_kw_sigma_replacer)
@all_of_type(ndarray)
@_get_docs
def hessian_matrix(image, sigma=1, mode='constant', cval=0, order='rc'):
    return (image, sigma)


@create_skimage_feature(_image_kw_sigma_replacer)
@all_of_type(ndarray)
@_get_docs
def hessian_matrix_det(image, sigma=1, approximate=True):
    return (image, sigma)


@create_skimage_feature(_Helems_replacer)
@all_of_type(ndarray)
@_get_docs
def hessian_matrix_eigvals(H_elems):
    return (H_elems,)


@create_skimage_feature(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def hog(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(3, 3),
    block_norm='L2-Hys',
    visualize=False,
    transform_sqrt=False,
    feature_vector=True,
    multichannel=None,
    *,
    channel_axis=None,
):
    return (image,)


@create_skimage_feature(_image_P_replacer)
@all_of_type(ndarray)
@_get_docs
def local_binary_pattern(image, P, R, method='default'):
    return (image, P)


@create_skimage_feature(_descriptors1_descriptors2_replacer)
@all_of_type(ndarray)
@_get_docs
def match_descriptors(
    descriptors1,
    descriptors2,
    metric=None,
    p=2,
    max_distance=np.inf,
    cross_check=True,
    max_ratio=1.0,
):
    return (descriptors1, descriptors2)


@create_skimage_feature(_image_template_replacer)
@all_of_type(ndarray)
@_get_docs
def match_template(
    image, template, pad_input=False, mode='constant', constant_values=0
):
    return (image, template)


@create_skimage_feature(_intimage_replacer)
@all_of_type(ndarray)
@_get_docs
def multiblock_lbp(int_image, r, c, width, height):
    return (int_image,)


@create_skimage_feature(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def multiscale_basic_features(
    image,
    multichannel=False,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=0.5,
    sigma_max=16,
    num_sigma=None,
    num_workers=None,
    *,
    channel_axis=None,
):
    return (image,)


@create_skimage_feature(
    _image_kw_mindistance_thresholdabs_thresholdrel_excludeborder_indices_numpeaks_footprint_labels_replacer
)
@all_of_type(ndarray)
@_get_docs
def peak_local_max(
    image,
    min_distance=1,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=True,
    indices=True,
    num_peaks=np.inf,
    footprint=None,
    labels=None,
    num_peaks_per_label=np.inf,
    p_norm=np.inf,
):
    return (image, footprint, labels)


@create_skimage_feature(
    _ax_image1_image2_keypoints1_keypoints2_matches_replacer
)
@all_of_type(ndarray)
@_get_docs
def plot_matches(
    ax,
    image1,
    image2,
    keypoints1,
    keypoints2,
    matches,
    keypoints_color='k',
    matches_color=None,
    only_matches=False,
    alignment='horizontal',
):
    return (image1, image2, keypoints1, keypoints2, matches)


@create_skimage_feature(_image_kw_sigma_replacer)
@all_of_type(ndarray)
@_get_docs
def shape_index(image, sigma=1, mode='constant', cval=0):
    return (image, sigma)


@create_skimage_feature(_image_kw_sigma_replacer)
@all_of_type(ndarray)
@_get_docs
def structure_tensor(image, sigma=1, mode='constant', cval=0, order=None):
    return (image, sigma)


@create_skimage_feature(_Aelems_replacer)
@all_of_type(ndarray)
@_get_docs
def structure_tensor_eigenvalues(A_elems):
    return (A_elems,)


@create_skimage_feature(_Axx_Axy_Ayy_replacer)
@all_of_type(ndarray)
@_get_docs
def structure_tensor_eigvals(Axx, Axy, Ayy):
    return (Axx, Axy, Ayy)
