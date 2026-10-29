from _skimage2.feature.template import match_template as match_template  # noqa: F401

__all__ = ['match_template']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
