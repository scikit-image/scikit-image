from _skimage2.io._plugins.gdal_plugin import imread as imread  # noqa: F401

__all__ = ['imread']

from skimage._docutils import adapt_doctests

adapt_doctests(globals())
