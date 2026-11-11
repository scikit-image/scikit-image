from _skimage2.feature.brief import (
    BRIEF as BRIEF,
    np2 as np2,
)  # noqa: F401

__all__ = [
    'BRIEF',
    'np2',
]

# Make the shim classes for the base classes first.
from . import util  # noqa: F401
from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
