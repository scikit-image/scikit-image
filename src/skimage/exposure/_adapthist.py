"""

Adapted from "Contrast Limited Adaptive Histogram Equalization" by Karel
Zuiderveld, Graphics Gems IV, Academic Press, 1994.

http://tog.acm.org/resources/GraphicsGems/

Relicensed with permission of the author under the Modified BSD license.

"""

from _skimage2.exposure._adapthist import (
    NR_OF_GRAY as NR_OF_GRAY,
    clip_histogram as clip_histogram,
    equalize_adapthist as equalize_adapthist,
    map_histogram as map_histogram,
)  # noqa: F401

__all__ = [
    'NR_OF_GRAY',
    'clip_histogram',
    'equalize_adapthist',
    'map_histogram',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
