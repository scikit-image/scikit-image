import pathlib
import warnings

import numpy as np
import imageio.v3 as iio

from .._shared.utils import warn, deprecate_parameter, DEPRECATED
from .._shared.version_requirements import require
from ..exposure import is_low_contrast
from ..color.colorconv import rgb2gray, rgba2rgb
from ..io.manage_plugins import call_plugin, plugin_order
from .util import file_or_url_context

from . import _deprecate_plugin_function

__all__ = [
    'imread',
    'imsave',
    'imshow',
    'show',
    'imread_collection',
    'imshow_collection',
]


@deprecate_parameter(
    deprecated_name="plugin",
    start_version="0.23",
    stop_version="0.25",
    template="Parameter `{deprecated_name}` is deprecated since version "
    "{deprecated_version} and will be removed in {changed_version} (or "
    "later). To avoid this warning, please do not use the parameter "
    "`{deprecated_name}`. Use imageio or other 3rd party libraries directly "
    "for more advanced IO features.",
)
def imread(fname, as_gray=False, plugin=DEPRECATED, **plugin_args):
    """Load an image from file.

    Parameters
    ----------
    fname : str or pathlib.Path
        Image file name, e.g. ``test.jpg`` or URL.
    as_gray : bool, optional
        If True, convert color images to gray-scale (64-bit floats).
        Images that are already in gray-scale format are not converted.

    Other Parameters
    ----------------
    **plugin_args : DEPRECATED
        `plugin_args` is deprecated.

        .. deprecated:: 0.23

    Returns
    -------
    img_array : ndarray
        The different color bands/channels are stored in the
        third dimension, such that a gray-image is MxN, an
        RGB-image MxNx3 and an RGBA-image MxNx4.

    """
    if plugin_args:
        warnings.warn(
            "Additional keyword arguments are deprecated since version "
            "0.23 and will be removed in 0.25 (or later). To avoid this "
            "warning, please do not use additional keyword arguments. "
            "Use imageio or a similar package instead.",
            category=FutureWarning,
            stacklevel=2,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message=".*Use imageio or a similar package instead.*",
            category=FutureWarning,
            module="skimage.io",
        )
        use_plugin = (
            plugin is not DEPRECATED
            or plugin_args
            or plugin_order()["imread"][0] != "imageio"
        )
        if use_plugin:
            plugin = None if plugin is DEPRECATED else plugin
            if isinstance(fname, pathlib.Path):
                fname = str(fname.resolve())

            if plugin is None and hasattr(fname, 'lower'):
                if fname.lower().endswith(('.tiff', '.tif')):
                    plugin = 'tifffile'

            with file_or_url_context(fname) as fname:
                img = call_plugin('imread', fname, plugin=plugin, **plugin_args)
        else:
            with file_or_url_context(fname) as fname:
                img = iio.imread(fname)

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


@_deprecate_plugin_function
def imread_collection(load_pattern, conserve_memory=True, plugin=None, **plugin_args):
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
    ic : :class:`ImageCollection`
        Collection of images.

    Other Parameters
    ----------------
    plugin_args : keywords
        Passed to the given plugin.

    """
    return call_plugin(
        'imread_collection', load_pattern, conserve_memory, plugin=plugin, **plugin_args
    )


@deprecate_parameter(
    deprecated_name="plugin",
    start_version="0.23",
    stop_version="0.25",
    template="Parameter `{deprecated_name}` is deprecated since version "
    "{deprecated_version} and will be removed in {changed_version} (or "
    "later). To avoid this warning, please do not use the parameter "
    "`{deprecated_name}`. Use imageio or other 3rd party libraries directly "
    "for more advanced IO features.",
)
def imsave(fname, arr, plugin=DEPRECATED, check_contrast=True, **plugin_args):
    """Save an image to file.

    Parameters
    ----------
    fname : str or pathlib.Path
        Target filename.
    arr : ndarray of shape (M,N) or (M,N,3) or (M,N,4)
        Image data.
    plugin : str, optional
        Name of plugin to use.  By default, the different plugins are
        tried (starting with imageio) until a suitable
        candidate is found.  If not given and fname is a tiff file, the
        tifffile plugin will be used.
    check_contrast : bool, optional
        Check for low contrast and print warning (default: True).

    Other Parameters
    ----------------
    plugin_args : keywords
        Passed to the given plugin.

    Notes
    -----
    When saving a JPEG, the compression ratio may be controlled using the
    ``quality`` keyword argument which is an integer with values in [1, 100],
    where 1 is worst quality and smallest file size, and 100 is the best quality
    and largest file size (default 75). This is only available when using
    the PIL and imageio plugins.
    """
    if plugin_args:
        warnings.warn(
            "Additional keyword arguments are deprecated since version "
            "0.23 and will be removed in 0.25 (or later). To avoid this "
            "warning, please do not use additional keyword arguments. "
            "Use imageio or a similar package instead.",
            category=FutureWarning,
            stacklevel=2,
        )

    if arr.dtype == bool:
        warn(
            f'{fname} is a boolean image: setting True to 255 and False to 0. '
            'To silence this warning, please convert the image using '
            'img_as_ubyte.',
            stacklevel=3,
        )
        arr = arr.astype('uint8') * 255
    if check_contrast and is_low_contrast(arr):
        warn(f'{fname} is a low contrast image', stacklevel=3)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message=".*Use imageio or a similar package instead.*",
            category=FutureWarning,
            module="skimage.io",
        )
        use_plugin = (
            plugin is not DEPRECATED
            or plugin_args
            or plugin_order()["imsave"][0] != "imageio"
        )
        if use_plugin:
            plugin = None if plugin is DEPRECATED else plugin
            if isinstance(fname, pathlib.Path):
                fname = str(fname.resolve())
            if plugin is None and hasattr(fname, 'lower'):
                if fname.lower().endswith(('.tiff', '.tif')):
                    plugin = 'tifffile'
            return call_plugin('imsave', fname, arr, plugin=plugin, **plugin_args)

    return iio.imwrite(fname, arr)


@_deprecate_plugin_function
def imshow(arr, plugin=None, **plugin_args):
    """Display an image.

    Parameters
    ----------
    arr : ndarray or str
        Image data or name of image file.
    plugin : str
        Name of plugin to use.  By default, the different plugins are
        tried (starting with imageio) until a suitable candidate is found.

    Other Parameters
    ----------------
    plugin_args : keywords
        Passed to the given plugin.

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(arr, str):
            arr = call_plugin('imread', arr, plugin=plugin)
        return call_plugin('imshow', arr, plugin=plugin, **plugin_args)


@_deprecate_plugin_function
def imshow_collection(ic, plugin=None, **plugin_args):
    """Display a collection of images.

    Parameters
    ----------
    ic : :class:`ImageCollection`
        Collection to display.
    plugin : str
        Name of plugin to use.  By default, the different plugins are
        tried until a suitable candidate is found.

    Other Parameters
    ----------------
    plugin_args : keywords
        Passed to the given plugin.

    """
    from ..io.manage_plugins import call_plugin

    return call_plugin('imshow_collection', ic, plugin=plugin, **plugin_args)


@_deprecate_plugin_function
@require("matplotlib", ">=3.3")
def show():
    """Display pending images.

    Launch the event loop of the current GUI plugin, and display all
    pending images, queued via `imshow`. This is required when using
    `imshow` from non-interactive scripts.

    A call to `show` will block execution of code until all windows
    have been closed.

    Examples
    --------
    .. testsetup::
        >>> import pytest; _ = pytest.importorskip('matplotlib')

    >>> import skimage.io as io
    >>> rng = np.random.default_rng()
    >>> for i in range(4):
    ...     ax_im = io.imshow(rng.random((50, 50)))
    >>> io.show() # doctest: +SKIP

    """
    return call_plugin('_app_show')
