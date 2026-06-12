from _skimage2.feature._hog import hog as hog  # noqa: F401

__all__ = ['hog']

from skimage._docutils import bind_namespace

bind_namespace(globals())
