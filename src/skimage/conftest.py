"""

This conftest is required to set the numpy print options
to legacy mode for doctests

"""

from _skimage2.conftest import handle_np2 as handle_np2  # noqa: F401

__all__ = ['handle_np2']
