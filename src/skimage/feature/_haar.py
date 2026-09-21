from _skimage2.feature._haar import (
    FEATURE_TYPE as FEATURE_TYPE,
    N_RECTANGLE as N_RECTANGLE,
    haar_like_feature_coord_wrapper as haar_like_feature_coord_wrapper,
    haar_like_feature_wrapper as haar_like_feature_wrapper,
)  # noqa: F401

__all__ = [
    'FEATURE_TYPE',
    'N_RECTANGLE',
    'haar_like_feature_coord_wrapper',
    'haar_like_feature_wrapper',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
