from _skimage2._shared.interpolation import coord_map_py as coord_map_py  # noqa: F401

__all__ = ['coord_map_py']

from skimage._docutils import bind_namespace

bind_namespace(globals())
