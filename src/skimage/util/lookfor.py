import sys

from _skimage2._vendored.numpy_lookfor import lookfor as _lookfor
from _skimage2.util._lookfor import __doc__  # noqa: F401


def lookfor(what):
    # Walk skimage public namespace; follow _skimage2 implementations only.
    return _lookfor(
        what,
        sys.modules['skimage'],
        namespace='skimage',
        other_namespaces=('_skimage2',),
    )
