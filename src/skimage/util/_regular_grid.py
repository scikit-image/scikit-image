from _skimage2.util._regular_grid import (
    regular_grid as regular_grid,
    regular_seeds as regular_seeds,
)  # noqa: F401

__all__ = [
    'regular_grid',
    'regular_seeds',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
