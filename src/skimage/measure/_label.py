from _skimage2.measure._label import label as label  # noqa: F401

__all__ = ['label']

from _skimage2.measure._label import _label_bool  # noqa: F401

from skimage._docutils import bind_namespace

bind_namespace(globals())
