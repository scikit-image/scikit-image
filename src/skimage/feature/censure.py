from _skimage2.feature.censure import (
    CENSURE as CENSURE,
    OCTAGON_INNER_SHAPE as OCTAGON_INNER_SHAPE,
    OCTAGON_OUTER_SHAPE as OCTAGON_OUTER_SHAPE,
    STAR_FILTER_SHAPE as STAR_FILTER_SHAPE,
    STAR_SHAPE as STAR_SHAPE,
)  # noqa: F401

__all__ = [
    'CENSURE',
    'OCTAGON_INNER_SHAPE',
    'OCTAGON_OUTER_SHAPE',
    'STAR_FILTER_SHAPE',
    'STAR_SHAPE',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
