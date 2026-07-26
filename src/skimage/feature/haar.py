from _skimage2.feature.haar import (
    FEATURE_TYPE as FEATURE_TYPE,
    draw_haar_like_feature as draw_haar_like_feature,
    haar_like_feature as haar_like_feature,
    haar_like_feature_coord as haar_like_feature_coord,
)  # noqa: F401

__all__ = [
    'FEATURE_TYPE',
    'draw_haar_like_feature',
    'haar_like_feature',
    'haar_like_feature_coord',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
