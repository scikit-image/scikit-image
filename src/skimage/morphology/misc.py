"""
Miscellaneous morphology functions.
"""

from _skimage2.morphology.misc import (
    funcs as funcs,
    remove_objects_by_distance as remove_objects_by_distance,
    remove_small_holes as remove_small_holes,
    remove_small_objects as remove_small_objects,
    skimage2ndimage as skimage2ndimage,
)  # noqa: F401

__all__ = [
    'funcs',
    'remove_objects_by_distance',
    'remove_small_holes',
    'remove_small_objects',
    'skimage2ndimage',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
