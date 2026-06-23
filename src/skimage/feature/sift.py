from _skimage2.feature.sift import SIFT as SIFT  # noqa: F401

__all__ = ['SIFT']

from skimage._docutils import adapt_doctests

adapt_doctests(globals())
