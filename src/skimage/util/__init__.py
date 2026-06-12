"""Utility functions to work with images in general."""

import lazy_loader as _lazy

_lazy_getattr, _, __all__ = _lazy.attach_stub(__name__, __file__)

# lazy_loader ignores the manually defined `__all__` in the PYI file. Instead,
# it populates `__all__` from what is imported. So patch in differences that we
# actually want to have in `__all__` and `__dir__`. For example, we want to keep
# deprecated functions available but not advertise them
# (see https://github.com/scientific-python/lazy-loader/pull/133)
to_strip = {
    # Allow backwards-compatible submodule access but don't advertise
    # or include in HTML docs
    'arraycrop',
    'compare',
    'dtype',
    'noise',
    'shape',
    'unique',
}
__all__ = list(set(__all__) - to_strip)
__all__ += ["PendingSkimage2Change", "FailedEstimationAccessError"]


def __dir__():
    return __all__.copy()


def __getattr__(name):
    obj = _lazy_getattr(name)

    if name == "lookfor":
        # Depending on how `lookfor` is first imported, lazy_loader may return
        # the module or the function of the same name. Avoid that and always
        # return the function.
        import importlib

        obj = importlib.import_module("skimage.util.lookfor").lookfor

    return obj


class PendingSkimage2Change(PendingDeprecationWarning):
    """Warning about API usage that will change when switching to ``skimage2``.

    As a subclass of :class:`PendingDeprecationWarning`, this warning isn't
    shown by default. But it can be enabled with a warnings filter to prepare
    for code changes related to skimage2 early on:

    .. code-block:: python

        import warnings
        from skimage.util import PendingSkimage2Change

        warnings.filterwarnings(
            action="default", category=PendingSkimage2Change
        )
    """


from _skimage2._shared.utils import FailedEstimationAccessError  # noqa: F401


# Bypass lazy_loader to maintain old behavior, that is, make the following pass:
#   from skimage.util.lookfor import lookfor
#   import skimage
#   assert callable(skimage.util.lookfor)
from .lookfor import lookfor  # noqa: F401
from .apply_parallel import apply_parallel  # noqa: F401
