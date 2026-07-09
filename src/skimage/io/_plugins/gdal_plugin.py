from _skimage2.io._plugins.gdal_plugin import imread as imread  # noqa: F401

__all__ = ['imread']

from skimage._docutils import bind_namespace

bind_namespace(globals())
