__doc__ = """Utilities to read and write images in various formats.

The following plug-ins are available:

"""

from ._plugins import *
from .sift import *
from .collection import *

from ._io import *
from ._image_stack import *
from .video import *


reset_plugins()


def _update_doc(doc):
    """Add a list of plugins to the module docstring, formatted as
    a ReStructuredText table.

    """
    from textwrap import wrap

    info = [(p, plugin_info(p)) for p in available_plugins if not p == 'test']

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
