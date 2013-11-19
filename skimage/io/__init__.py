"""Utilities to read and write images in various formats.

The following plug-ins are available:

"""

from ._plugins import *
from .sift import *
from .collection import *

from ._io import *
from ._image_stack import *
from .video import *


reset_plugins()

WRAP_LEN = 73


def _separator(char, lengths):
    return [char * separator_length for separator_length in lengths]


def _update_doc(doc):
    """Add a list of plugins to the module docstring, formatted as
    a ReStructuredText table.

    """
    from textwrap import wrap

    info = [(p, plugin_info(p).get('description', 'no description'))
            for p in available_plugins if not p == 'test']

    col_1_len = max([len(n) for (n, _) in info])
    col_2_len = WRAP_LEN - 1 - col_1_len

    # Insert table header
    info.insert(0, _separator('=', (col_1_len, col_2_len)))
    info.insert(1, ('Plugin', 'Description'))
    info.insert(2, _separator('-', (col_1_len, col_2_len)))
    info.append(_separator('-', (col_1_len, col_2_len)))

    for (name, plugin_description) in info:
        wrapped_descr = wrap(plugin_description, col_2_len)
        doc += "%s %s\n" % (name.ljust(col_1_len), '\n'.join(wrapped_descr))
    doc = doc.strip()

    return doc

__doc__ = _update_doc(__doc__)
