__all__ = ['imread', 'imread_collection', 'imsave', 'imshow', 'show',
           'push', 'pop']

from skimage.io._plugins import call as call_plugin
from skimage.color import rgb2grey
import numpy as np

# Shared image queue
_image_stack = []


def push(img):
    """Push an image onto the shared image stack.

    Parameters
    ----------
    img : ndarray
        Image to push.

    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Can only push ndarrays to the image stack.")

    _image_stack.append(img)


def pop():
    """Pop an image from the shared image stack.

    Returns
    -------
    img : ndarray
        Image popped from the stack.

    """
    return _image_stack.pop()


def imread(fname, as_grey=False, plugin=None, flatten=None,
           **plugin_args):
    """Load an image from file.

    Parameters
    ----------
    fname : string
        Image file name, e.g. ``test.jpg``.
    as_grey : bool
        If True, convert color images to grey-scale (32-bit floats).
        Images that are already in grey-scale format are not converted.
    plugin : str
        Name of plugin to use (Python Imaging Library by default).

    Other Parameters
    ----------------
    flatten : bool
        Backward compatible keyword, superseded by `as_grey`.

    Returns
    -------
    img_array : ndarray
        The different colour bands/channels are stored in the
        third dimension, such that a grey-image is MxN, an
        RGB-image MxNx3 and an RGBA-image MxNx4.

    Other parameters
    ----------------
    plugin_args : keywords
        Passed to the given plugin.

    """
    # Backward compatibility
    if flatten is not None:
        as_grey = flatten

    img = call_plugin('imread', fname, plugin=plugin, **plugin_args)

    if as_grey and getattr(img, 'ndim', 0) >= 3:
        img = rgb2grey(img)

    return img


def imread_collection(load_pattern, conserve_memory=True,
                      plugin=None, **plugin_args):
    """
    Load a collection of images.

    Parameters
    ----------
    load_pattern : str or list
        List of objects to load. These are usually filenames, but may
        vary depending on the currently active plugin.  See the docstring
        for ``ImageCollection`` for the default behaviour of this parameter.
    conserve_memory : bool, optional
        If True, never keep more than one in memory at a specific
        time.  Otherwise, images will be cached once they are loaded.

    Returns
    -------
    ic : ImageCollection
        Collection of images.

    Other parameters
    ----------------
    plugin_args : keywords
        Passed to the given plugin.

    """
    return call_plugin('imread_collection', load_pattern, conserve_memory,
                       plugin=plugin, **plugin_args)


def imsave(fname, arr, plugin=None, **plugin_args):
    """Save an image to file.

    Parameters
    ----------
    fname : str
        Target filename.
    arr : ndarray of shape (M,N) or (M,N,3) or (M,N,4)
        Image data.
    plugin : str
        Name of plugin to use.  By default, the different plugins are
        tried (starting with the Python Imaging Library) until a suitable
        candidate is found.

    Other parameters
    ----------------
    plugin_args : keywords
        Passed to the given plugin.

    """
    return call_plugin('imsave', fname, arr, plugin=plugin, **plugin_args)


def imshow(arr, plugin=None, **plugin_args):
    """Display an image.

    Parameters
    ----------
    arr : ndarray
        Image data.
    plugin : str
        Name of plugin to use.  By default, the different plugins are
        tried (starting with the Python Imaging Library) until a suitable
        candidate is found.

    Other parameters
    ----------------
    plugin_args : keywords
        Passed to the given plugin.

    """
    return call_plugin('imshow', arr, plugin=plugin, **plugin_args)


def show():
    '''Display pending images.

    Launch the event loop of the current gui plugin, and display all
    pending images, queued via `imshow`. This is required when using
    `imshow` from non-interactive scripts.

    A call to `show` will block execution of code until all windows
    have been closed.

    Examples
    --------
    >>> import skimage.io as io

    >>> for i in range(4):
    ...     io.imshow(np.random.random((50, 50)))
    >>> io.show()

    '''
    return call_plugin('_app_show')
