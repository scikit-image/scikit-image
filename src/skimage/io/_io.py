import pathlib

import numpy as np
import tifffile
import imageio.v3 as iio

from _skimage2._shared._warnings import warn_external
from ..exposure import is_low_contrast
from ..color.colorconv import rgb2gray, rgba2rgb
from .util import file_or_url_context
from .collection import ImageCollection

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

    Notes
    -----
    This function wraps :func:`imageio.v3.imread` and :func:`tifffile.imread`.
    The latter is only used if `fname` ends in ".tif" or ".tiff" (case ignored).
    Use the wrapped functions directly if you need to pass additional parameters.
    """
    if isinstance(fname, pathlib.Path):
        fname = str(fname.resolve())

    is_tiff_file = hasattr(fname, 'lower') and fname.lower().endswith(('.tiff', '.tif'))

    with file_or_url_context(fname) as fname:
        if is_tiff_file:
            img = tifffile.imread(fname)
        else:
            img = np.asarray(iio.imread(fname))
            if not img.flags['WRITEABLE']:
                img = img.copy()

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

    See Also
    --------
    skimage.io.ImageCollection
        The class, wrapped by this function.
    """
    return ImageCollection(
        load_pattern, conserve_memory=conserve_memory, load_func=iio.imread
    )


def _imsave_tiff(fname, arr):
    """Load a tiff image to file.

    Parameters
    ----------
    fname : str or file
        File name or file-like object.
    arr : ndarray
        The array to write.
    kwargs : keyword pairs, optional
        Additional keyword arguments to pass through (see ``tifffile``'s
        ``imwrite`` function).

    Notes
    -----
    Provided by the tifffile library [1]_, and supports many
    advanced image types including multi-page and floating-point.

    This implementation will set ``photometric='RGB'`` when writing if the first
    or last axis of `arr` has length 3 or 4. To override this, explicitly
    pass the ``photometric`` kwarg.

    This implementation will set ``planarconfig='SEPARATE'`` when writing if the
    first axis of arr has length 3 or 4. To override this, explicitly
    specify the ``planarconfig`` kwarg.

    References
    ----------
    .. [1] https://pypi.org/project/tifffile/

    """
    kwargs = {}
    if arr.shape[0] in [3, 4]:
        kwargs['planarconfig'] = 'SEPARATE'
        rgb = True
    else:
        rgb = arr.shape[-1] in [3, 4]
    if rgb:
        kwargs['photometric'] = 'RGB'

    return tifffile.imwrite(fname, arr, **kwargs)


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

    Notes
    -----
    This function is wraps :func:`imageio.v3.imwrite` and
    :func:`tifffile.imwrite`. The latter is only used if `fname` ends in ".tif"
    or ".tiff" (case ignored). Use the wrapped functions directly if you need
    to pass additional parameters.
    """
    if isinstance(fname, pathlib.Path):
        fname = str(fname.resolve())

    is_tiff_file = hasattr(fname, "lower") and fname.lower().endswith(('.tiff', '.tif'))

    if arr.dtype == bool:
        warn_external(
            f'{fname} is a boolean image: setting True to 255 and False to 0. '
            'To silence this warning, please convert the image using '
            'img_as_ubyte.',
        )
        arr = arr.astype('uint8') * 255
    if check_contrast and is_low_contrast(arr):
        warn_external(f'{fname} is a low contrast image')

    if is_tiff_file:
        return _imsave_tiff(fname, arr)
    else:
        return iio.imwrite(fname, arr)
