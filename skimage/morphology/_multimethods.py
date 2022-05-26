import functools
import warnings

import numpy as np
from numpy import dtype, ndarray
from uarray import generate_multimethod, Dispatchable
from uarray import all_of_type, create_multimethod

from skimage._backend import _mark_output, scalar_or_array
from . import _api


__all__ = [
    'area_closing',
    'area_opening',
    'ball',
    'binary_closing',
    'binary_dilation',
    'binary_erosion',
    'binary_opening',
    'black_tophat',
    'closing',
    'convex_hull_image',
    'convex_hull_object',
    'cube',
    'diameter_closing',
    'diameter_opening',
    'diamond',
    'dilation',
    'disk',
    'erosion',
    'flood',
    'flood_fill',
    'h_maxima',
    'h_minima',
    'label',
    'local_maxima',
    'local_minima',
    'max_tree',
    'max_tree_local_maxima',
    'medial_axis',
    'octagon',
    'octahedron',
    'opening',
    'reconstruction',
    'rectangle',
    'remove_small_holes',
    'remove_small_objects',
    'skeletonize',
    'skeletonize_3d',
    'square',
    'star',
    'thin',
    'white_tophat',
]


create_skimage_morphology = functools.partial(
    create_multimethod, domain="numpy.skimage.morphology"
)


_mark_scalar_or_array = functools.partial(
    Dispatchable, dispatch_type=scalar_or_array, coercible=True
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


def _image_kw_areathreshold_connectivity_parent_treetraverser_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        image,
        area_threshold=64,
        connectivity=1,
        parent=None,
        tree_traverser=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            area_threshold,
            connectivity,
            dispatchables[1],
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _radius_kw_dtype_replacer(args, kwargs, dispatchables):
    def self_method(radius, dtype='uint8', *args, **kwargs):
        kw_out = kwargs.copy()
        return (radius, dispatchables[0]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_footprint_out_replacer(args, kwargs, dispatchables):
    def self_method(image, footprint=None, out=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _width_kw_dtype_replacer(args, kwargs, dispatchables):
    def self_method(width, dtype='uint8', *args, **kwargs):
        kw_out = kwargs.copy()
        return (width, dispatchables[0]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_diameterthreshold_connectivity_parent_treetraverser_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        image,
        diameter_threshold=8,
        connectivity=1,
        parent=None,
        tree_traverser=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            diameter_threshold,
            connectivity,
            dispatchables[1],
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_seedpoint_kwonly_footprint_replacer(args, kwargs, dispatchables):
    def self_method(image, seed_point, **kwargs):
        kw_out = kwargs.copy()
        if 'footprint' in kw_out:
            kw_out['footprint'] = dispatchables[1]
        return (dispatchables[0], seed_point), kw_out

    return self_method(*args, **kwargs)


def _image_seedpoint_newvalue_kwonly_footprint_replacer(
    args, kwargs, dispatchables
):
    def self_method(image, seed_point, new_value, **kwargs):
        kw_out = kwargs.copy()
        if 'footprint' in kw_out:
            kw_out['footprint'] = dispatchables[1]
        return (dispatchables[0], seed_point, new_value), kw_out

    return self_method(*args, **kwargs)


def _image_h_kw_footprint_replacer(args, kwargs, dispatchables):
    def self_method(image, h, footprint=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], h, dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_footprint_replacer(args, kwargs, dispatchables):
    def self_method(image, footprint=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_connectivity_parent_treetraverser_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        image,
        connectivity=1,
        parent=None,
        tree_traverser=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            connectivity,
            dispatchables[1],
            dispatchables[2],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _image_kw_mask_replacer(args, kwargs, dispatchables):
    def self_method(image, mask=None, *args, **kwargs):
        kw_out = kwargs.copy()
        return (dispatchables[0], dispatchables[1]) + args, kw_out

    return self_method(*args, **kwargs)


def _m_n_kw_dtype_replacer(args, kwargs, dispatchables):
    def self_method(m, n, dtype='uint8', *args, **kwargs):
        kw_out = kwargs.copy()
        return (m, n, dispatchables[0]) + args, kw_out

    return self_method(*args, **kwargs)


def _seed_mask_kw_method_footprint_offset_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        seed,
        mask,
        method='dilation',
        footprint=None,
        offset=None,
        *args,
        **kwargs,
    ):
        kw_out = kwargs.copy()
        return (
            dispatchables[0],
            dispatchables[1],
            method,
            dispatchables[2],
            dispatchables[3],
        ) + args, kw_out

    return self_method(*args, **kwargs)


def _nrows_ncols_kw_dtype_replacer(args, kwargs, dispatchables):
    def self_method(nrows, ncols, dtype='uint8', *args, **kwargs):
        kw_out = kwargs.copy()
        return (nrows, ncols, dispatchables[0]) + args, kw_out

    return self_method(*args, **kwargs)


def _ar_kw_areathreshold_connectivity_kwonly_out_replacer(
    args, kwargs, dispatchables
):
    def self_method(
        ar, area_threshold=64, connectivity=1, **kwargs
    ):
        kw_out = kwargs.copy()
        if 'out' in kw_out:
            kw_out['out'] = dispatchables[1]
        return (dispatchables[0], area_threshold, connectivity), kw_out

    return self_method(*args, **kwargs)


def _ar_kw_minsize_connectivity_kwonly_out_replacer(
    args, kwargs, dispatchables
):
    def self_method(ar, min_size=64, connectivity=1, **kwargs):
        kw_out = kwargs.copy()
        if 'out' in kw_out:
            kw_out['out'] = dispatchables[1]
        return (dispatchables[0], min_size, connectivity), kw_out

    return self_method(*args, **kwargs)


def _a_kw_dtype_replacer(args, kwargs, dispatchables):
    def self_method(a, dtype='uint8', *args, **kwargs):
        kw_out = kwargs.copy()
        return (a, dispatchables[0]), kw_out

    return self_method(*args, **kwargs)


@create_skimage_morphology(
    _image_kw_areathreshold_connectivity_parent_treetraverser_replacer
)
@all_of_type(ndarray)
@_get_docs
def area_closing(
    image, area_threshold=64, connectivity=1, parent=None, tree_traverser=None
):
    return (image, parent, tree_traverser)


@create_skimage_morphology(
    _image_kw_areathreshold_connectivity_parent_treetraverser_replacer
)
@all_of_type(ndarray)
@_get_docs
def area_opening(
    image, area_threshold=64, connectivity=1, parent=None, tree_traverser=None
):
    return (image, parent, tree_traverser)


@create_skimage_morphology(_radius_kw_dtype_replacer)
@all_of_type(ndarray)
@_get_docs
def ball(radius, dtype='uint8'):
    return (Dispatchable(dtype, np.dtype),)


@create_skimage_morphology(_image_kw_footprint_out_replacer)
@all_of_type(ndarray)
@_get_docs
def binary_closing(image, footprint=None, out=None):
    return (image, footprint, _mark_output(out))


@create_skimage_morphology(_image_kw_footprint_out_replacer)
@all_of_type(ndarray)
@_get_docs
def binary_dilation(image, footprint=None, out=None):
    return (image, footprint, _mark_output(out))


@create_skimage_morphology(_image_kw_footprint_out_replacer)
@all_of_type(ndarray)
@_get_docs
def binary_erosion(image, footprint=None, out=None):
    return (image, footprint, _mark_output(out))


@create_skimage_morphology(_image_kw_footprint_out_replacer)
@all_of_type(ndarray)
@_get_docs
def binary_opening(image, footprint=None, out=None):
    return (image, footprint, _mark_output(out))


@create_skimage_morphology(_image_kw_footprint_out_replacer)
@all_of_type(ndarray)
@_get_docs
def black_tophat(image, footprint=None, out=None):
    return (image, footprint, _mark_output(out))


@create_skimage_morphology(_image_kw_footprint_out_replacer)
@all_of_type(ndarray)
@_get_docs
def closing(image, footprint=None, out=None):
    return (image, footprint, _mark_output(out))


@create_skimage_morphology(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def convex_hull_image(image, offset_coordinates=True, tolerance=1e-10):
    return (image,)


@create_skimage_morphology(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def convex_hull_object(image, *, connectivity=2):
    return (image,)


@create_skimage_morphology(_width_kw_dtype_replacer)
@all_of_type(ndarray)
@_get_docs
def cube(width, dtype='uint8', *, decomposition=None):
    return (Dispatchable(dtype, np.dtype),)


@create_skimage_morphology(
    _image_kw_diameterthreshold_connectivity_parent_treetraverser_replacer
)
@all_of_type(ndarray)
@_get_docs
def diameter_closing(
    image,
    diameter_threshold=8,
    connectivity=1,
    parent=None,
    tree_traverser=None,
):
    return (image, parent, tree_traverser)


@create_skimage_morphology(
    _image_kw_diameterthreshold_connectivity_parent_treetraverser_replacer
)
@all_of_type(ndarray)
@_get_docs
def diameter_opening(
    image,
    diameter_threshold=8,
    connectivity=1,
    parent=None,
    tree_traverser=None,
):
    return (image, parent, tree_traverser)


@create_skimage_morphology(_radius_kw_dtype_replacer)
@all_of_type(ndarray)
@_get_docs
def diamond(radius, dtype='uint8', *, decomposition=None):
    return (Dispatchable(dtype, np.dtype),)


@create_skimage_morphology(_image_kw_footprint_out_replacer)
@all_of_type(ndarray)
@_get_docs
def dilation(image, footprint=None, out=None, shift_x=False, shift_y=False):
    return (image, footprint, _mark_output(out))


@create_skimage_morphology(_radius_kw_dtype_replacer)
@all_of_type(ndarray)
@_get_docs
def disk(radius, dtype='uint8'):
    return (Dispatchable(dtype, np.dtype),)


@create_skimage_morphology(_image_kw_footprint_out_replacer)
@all_of_type(ndarray)
@_get_docs
def erosion(image, footprint=None, out=None, shift_x=False, shift_y=False):
    return (image, footprint, _mark_output(out))


@create_skimage_morphology(_image_seedpoint_kwonly_footprint_replacer)
@all_of_type(ndarray)
@_get_docs
def flood(
    image, seed_point, *, footprint=None, connectivity=None, tolerance=None
):
    return (image, footprint)


@create_skimage_morphology(_image_seedpoint_newvalue_kwonly_footprint_replacer)
@all_of_type(ndarray)
@_get_docs
def flood_fill(
    image,
    seed_point,
    new_value,
    *,
    footprint=None,
    connectivity=None,
    tolerance=None,
    in_place=False,
):
    return (image, footprint)


@create_skimage_morphology(_image_h_kw_footprint_replacer)
@all_of_type(ndarray)
@_get_docs
def h_maxima(image, h, footprint=None):
    return (image, footprint)


@create_skimage_morphology(_image_h_kw_footprint_replacer)
@all_of_type(ndarray)
@_get_docs
def h_minima(image, h, footprint=None):
    return (image, footprint)


@create_skimage_morphology(_image_kw_footprint_replacer)
@all_of_type(ndarray)
@_get_docs
def local_maxima(
    image, footprint=None, connectivity=None, indices=False, allow_borders=True
):
    return (image, footprint)


@create_skimage_morphology(_image_kw_footprint_replacer)
@all_of_type(ndarray)
@_get_docs
def local_minima(
    image, footprint=None, connectivity=None, indices=False, allow_borders=True
):
    return (image, footprint)


@create_skimage_morphology(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def max_tree(image, connectivity=1):
    return (image,)


@create_skimage_morphology(
    _image_kw_connectivity_parent_treetraverser_replacer
)
@all_of_type(ndarray)
@_get_docs
def max_tree_local_maxima(
    image, connectivity=1, parent=None, tree_traverser=None
):
    return (image, parent, tree_traverser)


@create_skimage_morphology(_image_kw_mask_replacer)
@all_of_type(ndarray)
@_get_docs
def medial_axis(image, mask=None, return_distance=False, *, random_state=None):
    return (image, mask)


@create_skimage_morphology(_m_n_kw_dtype_replacer)
@all_of_type(ndarray)
@_get_docs
def octagon(m, n, dtype='uint8', *, decomposition=None):
    return (Dispatchable(dtype, np.dtype),)


@create_skimage_morphology(_radius_kw_dtype_replacer)
@all_of_type(ndarray)
@_get_docs
def octahedron(radius, dtype='uint8', *, decomposition=None):
    return (Dispatchable(dtype, np.dtype),)


@create_skimage_morphology(_image_kw_footprint_out_replacer)
@all_of_type(ndarray)
@_get_docs
def opening(image, footprint=None, out=None):
    return (image, footprint, _mark_output(out))


@create_skimage_morphology(_seed_mask_kw_method_footprint_offset_replacer)
@all_of_type(ndarray)
@_get_docs
def reconstruction(seed, mask, method='dilation', footprint=None, offset=None):
    return (seed, mask, footprint, offset)


@create_skimage_morphology(_nrows_ncols_kw_dtype_replacer)
@all_of_type(ndarray)
@_get_docs
def rectangle(nrows, ncols, dtype='uint8', *, decomposition=None):
    return Dispatchable(dtype, np.dtype)


@create_skimage_morphology(
    _ar_kw_areathreshold_connectivity_kwonly_out_replacer
)
@all_of_type(ndarray)
@_get_docs
def remove_small_holes(
    ar, area_threshold=64, connectivity=1, *, out=None
):
    return (ar, _mark_output(out))


@create_skimage_morphology(
    _ar_kw_minsize_connectivity_kwonly_out_replacer
)
@all_of_type(ndarray)
@_get_docs
def remove_small_objects(
    ar, min_size=64, connectivity=1, *, out=None
):
    return (ar, _mark_output(out))


@create_skimage_morphology(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def skeletonize(image, *, method=None):
    return (image,)


@create_skimage_morphology(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def skeletonize_3d(image):
    return (image,)


@create_skimage_morphology(_width_kw_dtype_replacer)
@all_of_type(ndarray)
@_get_docs
def square(width, dtype='uint8', *, decomposition=None):
    return Dispatchable(dtype, np.dtype)


@create_skimage_morphology(_a_kw_dtype_replacer)
@all_of_type(ndarray)
@_get_docs
def star(a, dtype='uint8'):
    return Dispatchable(dtype, np.dtype)


@create_skimage_morphology(_image_replacer)
@all_of_type(ndarray)
@_get_docs
def thin(image, max_num_iter=None):
    return (image,)


@create_skimage_morphology(_image_kw_footprint_out_replacer)
@all_of_type(ndarray)
@_get_docs
def white_tophat(image, footprint=None, out=None):
    return (image, footprint, _mark_output(out))
