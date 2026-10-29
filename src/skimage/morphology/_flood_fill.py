"""
flood_fill.py - in place flood fill algorithm

This module provides a function to fill all equal (or within tolerance) values
connected to a given seed point with a different value.

"""

from _skimage2.morphology._flood_fill import (
    flood as flood,
    flood_fill as flood_fill,
)  # noqa: F401

__all__ = [
    'flood',
    'flood_fill',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
