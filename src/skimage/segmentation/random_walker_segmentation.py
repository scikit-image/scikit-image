"""

Random walker segmentation algorithm

from *Random walks for image segmentation*, Leo Grady, IEEE Trans
Pattern Anal Mach Intell. 2006 Nov;28(11):1768-83.

Installing pyamg and using the 'cg_mg' mode of random_walker improves
significantly the performance.

"""

from _skimage2.segmentation.random_walker_segmentation import (
    random_walker as random_walker,
)  # noqa: F401
from _skimage2.segmentation.random_walker_segmentation import UmfpackContext  # noqa: F401

__all__ = ['random_walker']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
