from _skimage2.segmentation.slic_superpixels import slic as slic  # noqa: F401

__all__ = ['slic']

from skimage._docutils import bind_namespace

bind_namespace(globals())
