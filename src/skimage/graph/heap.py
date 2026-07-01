"""

Cython implementation of a binary min heap.

"""

from _skimage2.graph.heap import (
    BinaryHeap as BinaryHeap,
    FastUpdateBinaryHeap as FastUpdateBinaryHeap,
)  # noqa: F401

__all__ = [
    'BinaryHeap',
    'FastUpdateBinaryHeap',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
