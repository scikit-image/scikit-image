__all__ = ['imread', 'imsave', 'imshow']

from scikits.image.io import plugin as _plugin

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
        Name of plugin to use.  By default, the different plugins are
        tried (starting with the Python Imaging Library) until a suitable
        candidate is found.

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

    return _plugin.call('read', fname, as_grey=as_grey, dtype=dtype,
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
    return _plugin.call('save', fname, arr, plugin=plugin, **plugin_args)

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
    return _plugin.call('show', arr, plugin=plugin, **plugin_args)
