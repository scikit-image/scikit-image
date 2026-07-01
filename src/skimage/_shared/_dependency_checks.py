from _skimage2._shared._dependency_checks import is_wasm as is_wasm  # noqa: F401

__all__ = ['is_wasm']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
