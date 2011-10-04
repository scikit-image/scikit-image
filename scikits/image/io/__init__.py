__doc__ = """Utilities to read and write images in various formats.

The following plug-ins are available:

"""

from _plugins import use as use_plugin
from _plugins import available as plugins
from _plugins import info as plugin_info

available_plugins = plugins()

for preferred_plugin in ['pil',
        'matplotlib', 'gtk', 'freeimage', 'qt', 'null']:
    if preferred_plugin in available_plugins:
        use_plugin(preferred_plugin)
        break

from sift import *
from collection import *

from io import *


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
