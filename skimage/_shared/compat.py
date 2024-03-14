"""Compatibility helpers for dependencies."""

from packaging.version import parse

import numpy as np


__all__ = [
    "NUMPY_LT_2_0_0",
    "NP_COPY_IF_NEEDED",
]


NUMPY_LT_2_0_0 = parse(np.__version__) < parse('2.0.0.dev0')

# With NumPy 2.0.0, `copy=False` now raises a ValueError if the copy cannot be
# made. The previous behavior to only copy if needed is provided with `copy=None`.
# During the transition period, use this symbol instead.
# Remove once NumPy 2.0.0 is the minimal required version.
# https://numpy.org/devdocs/release/2.0.0-notes.html#new-copy-keyword-meaning-for-array-and-asarray-constructors
# https://github.com/numpy/numpy/pull/25168
NP_COPY_IF_NEEDED = False if NUMPY_LT_2_0_0 else None
