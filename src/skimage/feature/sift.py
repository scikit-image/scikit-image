from _skimage2.feature.sift import SIFT as SIFT  # noqa: F401

__all__ = ['SIFT']

from skimage._docutils import bind_namespace

bind_namespace(globals())
