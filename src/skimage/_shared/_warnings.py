from _skimage2._shared._warnings import (
    all_warnings as all_warnings,
    expected_warnings as expected_warnings,
    warn as warn,
    warn_external as warn_external,
    ExperimentalAPIWarning as ExperimentalAPIWarning,
)  # noqa: F401

__all__ = [
    'all_warnings',
    'expected_warnings',
    'warn',
    'warn_external',
    'ExperimentalAPIWarning',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
