from _skimage2.transform.integral import (
    integral_image as integral_image,
    integrate as integrate,
)  # noqa: F401

__all__ = [
    'integral_image',
    'integrate',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
