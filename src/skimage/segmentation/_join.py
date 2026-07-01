from _skimage2.segmentation._join import (
    join_segmentations as join_segmentations,
    relabel_sequential as relabel_sequential,
)  # noqa: F401

__all__ = [
    'join_segmentations',
    'relabel_sequential',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
