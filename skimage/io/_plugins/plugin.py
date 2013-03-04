"""Handle image reading, writing and plotting plugins.

"""

__all__ = ['use', 'available', 'call', 'info', 'configuration', 'reset_plugins']

from ConfigParser import ConfigParser
import os.path
from glob import glob

plugin_store = None

plugin_provides = {}
plugin_module_name = {}
plugin_meta_data = {}


def reset_plugins():
    """Clear the plugin state to the default, i.e., where no plugins are loaded

    """
    global plugin_store
    plugin_store = {'imread': [],
                    'imsave': [],
                    'imshow': [],
                    'imread_collection': [],
                    '_app_show': []}

reset_plugins()


def _scan_plugins():
    """Scan the plugins directory for .ini files and parse them
    to gather plugin meta-data.

    """
    pd = os.path.dirname(__file__)
    ini = glob(os.path.join(pd, '*.ini'))

    for f in ini:
        cp = ConfigParser()
        cp.read(f)
        name = cp.sections()[0]

        meta_data = {}
        for opt in cp.options(name):
            meta_data[opt] = cp.get(name, opt)
        plugin_meta_data[name] = meta_data

        provides = [s.strip() for s in cp.get(name, 'provides').split(',')]
        valid_provides = [p for p in provides if p in plugin_store]

        for p in provides:
            if not p in plugin_store:
                print "Plugin `%s` wants to provide non-existent `%s`." \
                      " Ignoring." % (name, p)

        plugin_provides[name] = valid_provides
        plugin_module_name[name] = os.path.basename(f)[:-4]

_scan_plugins()


def call(kind, *args, **kwargs):
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


def use(name, kind=None):
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


def available(loaded=False):
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
    for plugin_func in plugin_store.itervalues():
        for plugin, func in plugin_func:
            active_plugins.add(plugin)

    d = {}
    for plugin in plugin_provides:
        if not loaded or plugin in active_plugins:
            d[plugin] = [f for f in plugin_provides[plugin] \
                         if not f.startswith('_')]

    return d


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
    if plugin in available(loaded=True):
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
            print "Plugin %s does not provide %s as advertised.  Ignoring." % \
                  (plugin, p)
        else:
            store = plugin_store[p]
            func = getattr(plugin_module, p)
            if not (plugin, func) in store:
                store.append((plugin, func))


def info(plugin):
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


def configuration():
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
