__all__ = ['imread', 'imsave', 'imshow', 'show', 'push', 'pop']

from scikits.image.io._plugins import call as call_plugin
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

def imread(fname, as_grey=False, dtype=None, plugin=None, flatten=None,
           **plugin_args):
    """Load an image from file.

    Parameters
    ----------
    fname : string
        Image file name, e.g. ``test.jpg``.
    as_grey : bool
        If True, convert color images to grey-scale. If `dtype` is not given,
        converted color images are returned as 32-bit float images.
        Images that are already in grey-scale format are not converted.
    dtype : dtype, optional
        NumPy data-type specifier. If given, the returned image has this type.
        If None (default), the data-type is determined automatically.
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

    return call_plugin('imread', fname, as_grey=as_grey, dtype=dtype,
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
    >>> import scikits.image.io as io

    >>> for i in range(4):
    ...     io.imshow(np.random.random((50, 50)))
    >>> io.show()

    '''
    return call_plugin('_app_show')
