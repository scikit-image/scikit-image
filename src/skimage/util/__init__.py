"""Utility functions to work with images in general."""

import lazy_loader as _lazy

__getattr__, _, __all__ = _lazy.attach_stub(__name__, __file__)

# lazy_loader ignores the manually defined `__all__` in the PYI file. Instead,
# it populates `__all__` from what is imported. So patch in differences that we
# actually want to have in `__all__` and `__dir__`. For example, we want to keep
# deprecated functions available but not advertise them
# (see https://github.com/scientific-python/lazy-loader/pull/133)
__all__ += ["PendingSkimage2Change", "FailedEstimationAccessError"]


def __dir__():
    return __all__.copy()


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
