from _skimage2.util.compare import compare_images as compare_images  # noqa: F401

__all__ = ['compare_images']

from skimage._docutils import bind_namespace

bind_namespace(globals())
