from _skimage2.filters._multiotsu import *  # noqa: F403
from _skimage2.filters._multiotsu import __doc__  # noqa: F401
from _skimage2.filters._multiotsu import (  # noqa: F401
    _get_multiotsu_thresh_indices,
    _get_multiotsu_thresh_indices_lut,
)

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
