from _skimage2.measure.entropy import shannon_entropy as shannon_entropy  # noqa: F401

__all__ = ['shannon_entropy']

from skimage._docutils import bind_namespace

bind_namespace(globals())
