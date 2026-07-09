from _skimage2.data._registry import (
    registry as registry,
    registry_urls as registry_urls,
)  # noqa: F401

__all__ = [
    'registry',
    'registry_urls',
]

from skimage._docutils import bind_namespace

bind_namespace(globals())
