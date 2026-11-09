"""
Utility functions used in the morphology subpackage.
"""

from _skimage2.morphology._util import *  # noqa: F403
from _skimage2.morphology._util import __doc__  # noqa: F401
from _skimage2.morphology._util import _offsets_to_raveled_neighbors  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
