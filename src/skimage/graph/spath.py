from _skimage2.graph.spath import shortest_path as shortest_path  # noqa: F401

__all__ = ['shortest_path']

from skimage._docutils import bind_namespace

bind_namespace(globals())
