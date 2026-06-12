"""
watershed.pyx - cython implementation of guts of watershed

"""

from _skimage2.segmentation._watershed_cy import watershed_raveled as watershed_raveled  # noqa: F401

__all__ = ['watershed_raveled']
