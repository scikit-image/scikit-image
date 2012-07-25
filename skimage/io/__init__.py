__doc__ = """Utilities to read and write images in various formats.

The following plug-ins are available:

"""

from ._plugins import use as use_plugin
from ._plugins import available as plugins
from ._plugins import info as plugin_info
from ._plugins import configuration as plugin_order
from ._plugins import reset_plugins as _reset_plugins

from .sift import *
from .collection import *

from ._io import *
from .video import *


available_plugins = plugins()


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
    _reset_plugins()
    _load_preferred_plugins()

def _update_doc(doc):
    """Add a list of plugins to the module docstring, formatted as
    a ReStructuredText table.

    """
    from textwrap import wrap

    info = [(p, plugin_info(p)) for p in plugins() if not p == 'test']
    col_1_len = max([len(n) for (n, _) in info])

    wrap_len = 73
    col_2_len = wrap_len - 1 - col_1_len

    # Insert table header
    info.insert(0, ('=' * col_1_len, {'description': '=' * col_2_len}))
    info.insert(1, ('Plugin', {'description': 'Description'}))
    info.insert(2, ('-' * col_1_len, {'description': '-' * col_2_len}))
    info.append(('=' * col_1_len, {'description': '=' * col_2_len}))

    for (name, meta_data) in info:
        wrapped_descr = wrap(meta_data.get('description', ''),
                             col_2_len)
        doc += "%s %s\n" % (name.ljust(col_1_len),
                            '\n'.join(wrapped_descr))
    doc = doc.strip()

    return doc

__doc__ = _update_doc(__doc__)

reset_plugins()
