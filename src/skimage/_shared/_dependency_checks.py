from _skimage2._shared._dependency_checks import is_wasm as is_wasm  # noqa: F401

__all__ = ['is_wasm']

from skimage._docutils import bind_namespace

bind_namespace(globals())
