"""Handle image reading, writing and plotting plugins.

"""

try:
    from configparser import ConfigParser  # Python 3
except ImportError:
    from ConfigParser import ConfigParser  # Python 2

import os.path
from glob import glob


__all__ = ['use_plugin', 'call_plugin', 'plugin_info', 'plugin_order',
           'reset_plugins', 'find_available_plugins', 'available_plugins']


plugin_store = None
plugin_provides = {}
plugin_module_name = {}
plugin_meta_data = {}


def _clear_plugins():
    """Clear the plugin state to the default, i.e., where no plugins are loaded

    """
    global plugin_store
    plugin_store = {'imread': [],
                    'imsave': [],
                    'imshow': [],
                    'imread_collection': [],
                    '_app_show': []}
_clear_plugins()


def _load_preferred_plugins():
    # Load preferred plugin for each io function.
    io_funcs = ['imsave', 'imshow', 'imread_collection', 'imread']
    preferred_plugins = ['matplotlib', 'pil', 'qt', 'freeimage', 'null']
    for func in io_funcs:
        for plugin in preferred_plugins:
            if plugin not in available_plugins:
                continue
            try:
                use_plugin(plugin, kind=func)
                break
            except (ImportError, RuntimeError, OSError):
                pass

    # Use PIL as the default imread plugin, since matplotlib (1.2.x)
    # is buggy (flips PNGs around, returns bytes as floats, etc.)
    try:
        use_plugin('pil', 'imread')
    except ImportError:
        pass


def reset_plugins():
    _clear_plugins()
    _load_preferred_plugins()


def _parse_config_file(filename):
    """Return plugin name and meta-data dict from plugin config file."""
    parser = ConfigParser()
    parser.read(filename)
    name = parser.sections()[0]

    meta_data = {}
    for opt in parser.options(name):
        meta_data[opt] = parser.get(name, opt)

    return name, meta_data


def _scan_plugins():
    """Scan the plugins directory for .ini files and parse them
    to gather plugin meta-data.

    """
    pd = os.path.dirname(__file__)
    config_files = glob(os.path.join(pd, '*.ini'))

    for filename in config_files:
        name, meta_data = _parse_config_file(filename)
        plugin_meta_data[name] = meta_data

        provides = [s.strip() for s in meta_data['provides'].split(',')]
        valid_provides = [p for p in provides if p in plugin_store]

        for p in provides:
            if not p in plugin_store:
                print("Plugin `%s` wants to provide non-existent `%s`." \
                      " Ignoring." % (name, p))

        plugin_provides[name] = valid_provides
        plugin_module_name[name] = os.path.basename(filename)[:-4]

_scan_plugins()


def find_available_plugins(loaded=False):
    """List available plugins.

    Parameters
    ----------
    loaded : bool
        If True, show only those plugins currently loaded.  By default,
        all plugins are shown.

    Returns
    -------
    p : dict
        Dictionary with plugin names as keys and exposed functions as
        values.

    """
    active_plugins = set()
    for plugin_func in plugin_store.values():
        for plugin, func in plugin_func:
            active_plugins.add(plugin)

    d = {}
    for plugin in plugin_provides:
        if not loaded or plugin in active_plugins:
            d[plugin] = [f for f in plugin_provides[plugin] \
                         if not f.startswith('_')]

    return d


available_plugins = find_available_plugins()


def call_plugin(kind, *args, **kwargs):
    """Find the appropriate plugin of 'kind' and execute it.

    Parameters
    ----------
    kind : {'imshow', 'imsave', 'imread', 'imread_collection'}
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

You may load I/O plugins with the `skimage.io.use_plugin`
command.  A list of all available plugins can be found using
`skimage.io.plugins()`.''' % kind)

    plugin = kwargs.pop('plugin', None)
    if plugin is None:
        _, func = plugin_funcs[0]
    else:
        _load(plugin)
        try:
            func = [f for (p, f) in plugin_funcs if p == plugin][0]
        except IndexError:
            raise RuntimeError('Could not find the plugin "%s" for %s.' % \
                               (plugin, kind))

    return func(*args, **kwargs)


def use_plugin(name, kind=None):
    """Set the default plugin for a specified operation.  The plugin
    will be loaded if it hasn't been already.

    Parameters
    ----------
    name : str
        Name of plugin.
    kind : {'imsave', 'imread', 'imshow', 'imread_collection'}, optional
        Set the plugin for this function.  By default,
        the plugin is set for all functions.

    See Also
    --------
    plugins : List of available plugins

    Examples
    --------

    Use the Python Imaging Library to read images:

    >>> from skimage.io import use_plugin
    >>> use_plugin('pil', 'imread')

    """
    if kind is None:
        kind = plugin_store.keys()
    else:
        if not kind in plugin_provides[name]:
            raise RuntimeError("Plugin %s does not support `%s`." % \
                               (name, kind))

        if kind == 'imshow':
            kind = [kind, '_app_show']
        else:
            kind = [kind]

    _load(name)

    for k in kind:
        if not k in plugin_store:
            raise RuntimeError("'%s' is not a known plugin function." % k)

        funcs = plugin_store[k]

        # Shuffle the plugins so that the requested plugin stands first
        # in line
        funcs = [(n, f) for (n, f) in funcs if n == name] + \
                [(n, f) for (n, f) in funcs if n != name]

        plugin_store[k] = funcs



def _load(plugin):
    """Load the given plugin.

    Parameters
    ----------
    plugin : str
        Name of plugin to load.

    See Also
    --------
    plugins : List of available plugins

    """
    if plugin in find_available_plugins(loaded=True):
        return
    if not plugin in plugin_module_name:
        raise ValueError("Plugin %s not found." % plugin)
    else:
        modname = plugin_module_name[plugin]
        plugin_module = __import__('skimage.io._plugins.' + modname,
                                   fromlist=[modname])

    provides = plugin_provides[plugin]
    for p in provides:
        if not hasattr(plugin_module, p):
            print("Plugin %s does not provide %s as advertised.  Ignoring." % \
                  (plugin, p))
        else:
            store = plugin_store[p]
            func = getattr(plugin_module, p)
            if not (plugin, func) in store:
                store.append((plugin, func))


def plugin_info(plugin):
    """Return plugin meta-data.

    Parameters
    ----------
    plugin : str
        Name of plugin.

    Returns
    -------
    m : dict
        Meta data as specified in plugin ``.ini``.

    """
    try:
        return plugin_meta_data[plugin]
    except KeyError:
        raise ValueError('No information on plugin "%s"' % plugin)


def plugin_order():
    """Return the currently preferred plugin order.

    Returns
    -------
    p : dict
        Dictionary of preferred plugin order, with function name as key and
        plugins (in order of preference) as value.

    """
    p = {}
    for func in plugin_store:
        p[func] = [plugin_name for (plugin_name, f) in plugin_store[func]]
    return p
