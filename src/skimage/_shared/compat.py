"""
Compatibility helpers for dependencies.
"""

from _skimage2._shared.compat import (
    NP_COPY_IF_NEEDED as NP_COPY_IF_NEEDED,
    SCIPY_CG_TOL_PARAM_NAME as SCIPY_CG_TOL_PARAM_NAME,
)  # noqa: F401

__all__ = [
    'NP_COPY_IF_NEEDED',
    'SCIPY_CG_TOL_PARAM_NAME',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
