from _skimage2._shared.interpolation import coord_map_py as coord_map_py  # noqa: F401

__all__ = ['coord_map_py']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
