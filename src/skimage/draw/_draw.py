from _skimage2.draw._draw import *  # noqa: F403
from _skimage2.draw._draw import __doc__  # noqa: F401
from _skimage2.draw._draw import _bezier_segment  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
