from _skimage2.filters._window import window as window  # noqa: F401

__all__ = ['window']

from skimage._docutils import bind_namespace

bind_namespace(globals())
