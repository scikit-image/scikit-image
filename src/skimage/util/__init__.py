"""Utility functions to work with images in general."""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)

from _skimage2._shared.utils import FailedEstimationAccessError  # noqa: F401


class PendingSkimage2Change(PendingDeprecationWarning):
    """A warning about API usage that will silently change or break in skimage2.

    As a subclass of :class:`PendingDeprecationWarning`, this warning isn't
    shown by default. But it can be enabled with a warnings filter to prepare
    for code changes related to skimage2 early on:

    .. code-block:: python

        import warnings
        import _skimage2 as ski2
        warnings.filterwarnings(
            action="default", category=ski.util.PendingSkimage2Change
        )
    """
