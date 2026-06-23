from _skimage2.feature.brief import (
    BRIEF as BRIEF,
    np2 as np2,
)  # noqa: F401

__all__ = [
    'BRIEF',
    'np2',
]

from skimage._docutils import adapt_doctests

adapt_doctests(globals())
