import pathlib

import numpy as np

from _skimage2._shared._warnings import warn_external
from ..exposure import is_low_contrast
from ..color.colorconv import rgb2gray, rgba2rgb
from ..io.manage_plugins import call_plugin, _hide_plugin_deprecation_warnings
from .util import file_or_url_context

__all__ = [
    'imread',
    'imsave',
    'imread_collection',
]


def imread(fname, as_gray=False):
    """Load an image from file.

    Parameters
    ----------
    fname : str or pathlib.Path
        Image file name, e.g. ``test.jpg`` or URL.
    as_gray : bool, optional
        If True, convert color images to gray-scale (64-bit floats).
        Images that are already in gray-scale format are not converted.

    Returns
    -------
    img_array : ndarray
        The different color bands/channels are stored in the
        third dimension, such that a gray-image is MxN, an
        RGB-image MxNx3 and an RGBA-image MxNx4.
    """
    plugin = None

    if isinstance(fname, pathlib.Path):
        fname = str(fname.resolve())

    if hasattr(fname, 'lower'):
        if fname.lower().endswith(('.tiff', '.tif')):
            plugin = 'tifffile'

    with file_or_url_context(fname) as fname, _hide_plugin_deprecation_warnings():
        img = call_plugin('imread', fname, plugin=plugin)

    if not hasattr(img, 'ndim'):
        return img

    if img.ndim > 2:
        if img.shape[-1] not in (3, 4) and img.shape[-3] in (3, 4):
            img = np.swapaxes(img, -1, -3)
            img = np.swapaxes(img, -2, -3)

        if as_gray:
            if img.shape[2] == 4:
                img = rgba2rgb(img)
            img = rgb2gray(img)

    return img


def imread_collection(load_pattern, conserve_memory=True):
    """
    Load a collection of images.

    Parameters
    ----------
    load_pattern : str or list
        List of objects to load. These are usually filenames, but may
        vary depending on the currently active plugin. See :class:`ImageCollection`
        for the default behaviour of this parameter.
    conserve_memory : bool, optional
        If True, never keep more than one in memory at a specific
        time.  Otherwise, images will be cached once they are loaded.

    Returns
    -------
    ic : :class:`~.ImageCollection`
        Collection of images.
    """
    plugin = None
    with _hide_plugin_deprecation_warnings():
        return call_plugin(
            'imread_collection',
            load_pattern,
            conserve_memory,
            plugin=plugin,
        )


def imsave(fname, arr, *, check_contrast=True):
    """Save an image to file.

    Parameters
    ----------
    fname : str or pathlib.Path
        Target filename.
    arr : ndarray of shape (M, N[, C]), with C=3 or C=4
        Image data.
    check_contrast : bool, optional
        Check for low contrast and print warning (default: True).
    """
    plugin = None

    if isinstance(fname, pathlib.Path):
        fname = str(fname.resolve())
    if plugin is None and hasattr(fname, 'lower'):
        if fname.lower().endswith(('.tiff', '.tif')):
            plugin = 'tifffile'
    if arr.dtype == bool:
        warn_external(
            f'{fname} is a boolean image: setting True to 255 and False to 0. '
            'To silence this warning, please convert the image using '
            'img_as_ubyte.',
        )
        arr = arr.astype('uint8') * 255
    if check_contrast and is_low_contrast(arr):
        warn_external(f'{fname} is a low contrast image')

    with _hide_plugin_deprecation_warnings():
        return call_plugin('imsave', fname, arr, plugin=plugin)
