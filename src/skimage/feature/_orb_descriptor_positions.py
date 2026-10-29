from _skimage2.feature._orb_descriptor_positions import (
    POS as POS,
    POS0 as POS0,
    POS1 as POS1,
    this_dir as this_dir,
)  # noqa: F401

__all__ = [
    'POS',
    'POS0',
    'POS1',
    'this_dir',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
