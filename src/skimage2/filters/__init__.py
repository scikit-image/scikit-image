"""Sharpening, edge finding, rank filters, thresholding, etc."""

import lazy_loader as _lazy

from _skimage2 import filters as _impl

__dir__ = _impl.__dir__
__all__ = _impl.__all__

# `rank` must resolve to the shim subpackage, so that `from skimage2.filters
# import rank` and `import skimage2.filters.rank` yield the same module.
_submodule_getattr, _, _ = _lazy.attach(__name__, submodules=["rank"])


def __getattr__(name):
    if name == "rank":
        return _submodule_getattr(name)
    return _impl.__getattr__(name)
