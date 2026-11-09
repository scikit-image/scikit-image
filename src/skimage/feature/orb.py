from _skimage2.feature.orb import (
    OFAST_MASK as OFAST_MASK,
    OFAST_UMAX as OFAST_UMAX,
    ORB as ORB,
)  # noqa: F401

__all__ = [
    'OFAST_MASK',
    'OFAST_UMAX',
    'ORB',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
