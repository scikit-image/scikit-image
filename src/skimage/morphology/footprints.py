from _skimage2.morphology.footprints import (
    ball as ball,
    cube as cube,
    diamond as diamond,
    disk as disk,
    ellipse as ellipse,
    footprint_from_sequence as footprint_from_sequence,
    footprint_rectangle as footprint_rectangle,
    octagon as octagon,
    octahedron as octahedron,
    rectangle as rectangle,
    square as square,
    star as star,
)  # noqa: F401

__all__ = [
    'ball',
    'cube',
    'diamond',
    'disk',
    'ellipse',
    'footprint_from_sequence',
    'footprint_rectangle',
    'octagon',
    'octahedron',
    'rectangle',
    'square',
    'star',
]

from _skimage2.morphology._footprints import mirror_footprint, pad_footprint  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
