"""Morphological algorithms, e.g., closing, opening, skeletonization."""

import lazy_loader as _lazy

__getattr__, _, __all__ = _lazy.attach_stub(__name__, __file__)

# lazy_loader ignores the manually defined `__all__` in the PYI file. Instead,
# it populates `__all__` from what is imported. So patch in differences that we
# actually want to have in `__all__` and `__dir__`. For example, we want to keep
# deprecated functions available but not advertise them
# (see https://github.com/scientific-python/lazy-loader/pull/133)
to_strip = {
    # Deprecated
    'binary_closing',
    'binary_dilation',
    'binary_erosion',
    'binary_opening',
    'cube',
    'rectangle',
    'square',
    # Allow backwards-compatible submodule access but don't advertise
    # or include in HTML docs
    'binary',
    'convex_hull',
    'extrema',
    'footprints',
    'gray',
    'grayreconstruct',
    'isotropic',
    'misc',
}
__all__ = list(set(__all__) - to_strip)

# Lazy loader can only include submodules, so include `label` manually
__all__.append("label")


def __dir__():
    return __all__.copy()


from ..measure._label import label  # noqa: F401

# Bypass lazy_loader to maintain old behavior, that is, make the following pass:
#   from skimage.morphology.max_tree import max_tree
#   import skimage
#   assert callable(skimage.morphology.max_tree)
from .max_tree import max_tree  # noqa: F401
