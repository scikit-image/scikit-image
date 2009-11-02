"""Handle image reading, writing and plotting plugins.

"""

__all__ = ['register', 'use', 'load', 'available', 'call']

import warnings

plugin_store = {'read': [],
                'save': [],
                'show': [],
                'appshow': []}

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

        plugin_store[kind].insert(0, (name, func))


def call(kind, *args, **kwargs):
    """Find the appropriate plugin of 'kind' and execute it.

    Parameters
    ----------
    kind : {'show', 'save', 'read'}
        Function to look up.
    plugin : str, optional
        Plugin to load.  Defaults to None, in which case the first
        matching plugin is used.
    *args, **kwargs : arguments and keyword arguments
        Passed to the plugin function.

    """
    if not kind in plugin_store:
        raise ValueError('Invalid function (%s) requested.' % kind)

    plugin_funcs = plugin_store[kind]
    if len(plugin_funcs) == 0:
        raise RuntimeError('''No suitable plugin registered for %s.

You may load I/O plugins with the `scikits.image.io.load_plugin`
command.  A list of all available plugins can be found using
`scikits.image.io.plugins()`.''' % kind)

    plugin = kwargs.pop('plugin', None)
    if plugin is None:
        _, func = plugin_funcs[0]
    else:
        try:
            func = [f for (p,f) in plugin_funcs if p == plugin][0]
        except IndexError:
            raise RuntimeError('Could not find the plugin "%s" for %s.' % \
                               (plugin, kind))

    return func(*args, **kwargs)

def use(name, kind=None):
    """Set the default plugin for a specified operation.

    Parameters
    ----------
    name : str
        Name of plugin.
    kind : {'save', 'read', 'show'}, optional
        Set the plugin for this function.  By default,
        the plugin is set for all functions.

    Examples
    --------

    Use Python Imaging Library to read images:

    >>> from scikits.image.io import plugin
    >>> plugin.use('PIL', 'read')

    """
    if kind is None:
        kind = plugin_store.keys()
    else:
        kind = [kind]

    for k in kind:
        if not k in plugin_store:
            raise RuntimeError("Could not find plugin for '%s'" % k)

        funcs = plugin_store[k]

        # Shuffle the plugins so that the requested plugin stands first
        # in line
        funcs = [(n, f) for (n, f) in funcs if n == name] + \
                [(n, f) for (n, f) in funcs if n != name]

        n, f = funcs[0]
        if not n == name:
            warnings.warn(RuntimeWarning('Could not set plugin "%s" for'
                                         ' function "%s".' % (name, k)))

        plugin_store[k] = funcs

def available(kind=None):
    """List available plugins.

    Parameters
    ----------
    kind : {'show', 'save', 'read'}, optional
        Display the plugin list for the given function type.  If not
        specified, return a dictionary with the plugins for all
        functions.

    """
    if kind is None:
        kind = plugin_store.keys()
    else:
        kind = [kind]

    d = {}
    for k in kind:
        if not k in plugin_store:
            raise ValueError('No function "%s" exists in the plugin registry.'
                             % kind)

        d[k] = [name for (name, func) in plugin_store[k]]

    return d

def load(plugin):
    """Load the given plugin.

    Parameters
    ----------
    plugin : str
        Name of plugin to load.

    See Also
    --------
    plugins : List of available plugins

    """
    try:
        __import__('scikits.image.io._plugins.' + plugin + "_plugin")
    except ImportError:
        raise ValueError('Plugin %s not found.' % plugin)
