"""
watershed.py - watershed algorithm

This module implements a watershed algorithm that apportions pixels into
marked basins. The algorithm uses a priority queue to hold the pixels
with the metric for the priority queue being pixel value, then the time
of entry into the queue - this settles ties in favor of the closest marker.

Some ideas taken from
Soille, "Automated Basin Delineation from Digital Elevation Models Using
Mathematical Morphology", Signal Processing 20 (1990) 171-182.

The most important insight in the paper is that entry time onto the queue
solves two problems: a pixel should be assigned to the neighbor with the
largest gradient or, if there is no gradient, pixels on a plateau should
be split between markers on opposite sides.

"""

from _skimage2.segmentation._watershed import watershed as watershed  # noqa: F401
from _skimage2.morphology._flood_fill import flood as flood  # noqa: F401
from _skimage2.morphology._flood_fill import flood_fill as flood_fill  # noqa: F401

__all__ = ['watershed']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
