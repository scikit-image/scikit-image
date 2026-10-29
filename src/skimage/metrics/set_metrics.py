from _skimage2.metrics.set_metrics import (
    hausdorff_distance as hausdorff_distance,
    hausdorff_pair as hausdorff_pair,
)  # noqa: F401

__all__ = [
    'hausdorff_distance',
    'hausdorff_pair',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
