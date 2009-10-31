"""Handle image reading, writing and plotting plugins.

"""

plugin_store = {'read': [],
                'save': [],
                'show': []}

def register(name, **kwds):
    """Register an image I/O plugin.

    Parameters
    ----------
    name : str
        Name of this plugin.
    read : callable, optional
        Function with signature
        ``read(filename, as_grey=False, dtype=None, **plugin_specific_args)``
        that reads images.
    save : callable, optional
        Function with signature
        ``write(filename, arr, **plugin_specific_args)``
        that writes an image to disk.
    show : callable, optional
        Function with signature
        ``show(X, **plugin_specific_args)`` that displays an image.

    """
    for kind in kwds:
        if kind not in plugin_store.keys():
            raise ValueError('Tried to register invalid plugin method.')

        func = kwds[kind]
        if not callable(func):
            raise ValueError('Can only register functions as plugins.')

        plugin_store[kind].append((name, func))
