from _skimage2._shared._tempfile import temporary_file as temporary_file  # noqa: F401

__all__ = ['temporary_file']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
