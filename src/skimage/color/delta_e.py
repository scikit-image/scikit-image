"""

Functions for calculating the "distance" between colors.

Implicit in these definitions of "distance" is the notion of "Just Noticeable
Distance" (JND).  This represents the distance between colors where a human can
perceive different colors.  Humans are more sensitive to certain colors than
others, which different deltaE metrics correct for with varying degrees of
sophistication.

The literature often mentions 1 as the minimum distance for visual
differentiation, but more recent studies (Mahy 1994) peg JND at 2.3

The delta-E notation comes from the German word for "Sensation" (Empfindung).

References
----------
.. [1] https://en.wikipedia.org/wiki/Color_difference


"""

from _skimage2.color.delta_e import (
    deltaE_cie76 as deltaE_cie76,
    deltaE_ciede2000 as deltaE_ciede2000,
    deltaE_ciede94 as deltaE_ciede94,
    deltaE_cmc as deltaE_cmc,
    get_dH2 as get_dH2,
)  # noqa: F401

__all__ = [
    'deltaE_cie76',
    'deltaE_ciede2000',
    'deltaE_ciede94',
    'deltaE_cmc',
    'get_dH2',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
