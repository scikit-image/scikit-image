from _skimage2.data._registry import (
    registry as registry,
    registry_urls as registry_urls,
)  # noqa: F401

__all__ = [
    'registry',
    'registry_urls',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
