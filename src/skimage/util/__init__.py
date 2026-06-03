"""Utility functions to work with images in general."""

import lazy_loader as _lazy


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


__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)

from _skimage2._shared.utils import FailedEstimationAccessError  # noqa: F401
