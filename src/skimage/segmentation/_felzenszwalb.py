from _skimage2.segmentation._felzenszwalb import felzenszwalb as felzenszwalb  # noqa: F401

__all__ = ['felzenszwalb']

from skimage._docutils import bind_namespace

bind_namespace(globals())
