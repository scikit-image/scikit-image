from _skimage2.restoration.unwrap import unwrap_phase as unwrap_phase  # noqa: F401

__all__ = ['unwrap_phase']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
