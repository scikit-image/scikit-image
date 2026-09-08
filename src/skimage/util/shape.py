from _skimage2.util.shape import (
    view_as_blocks as view_as_blocks,
    view_as_windows as view_as_windows,
)  # noqa: F401

__all__ = [
    'view_as_blocks',
    'view_as_windows',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
