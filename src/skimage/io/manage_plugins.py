"""Shim: delegate to ``_skimage2.io.manage_plugins``."""

import sys

from _skimage2.io import manage_plugins

sys.modules[__name__] = manage_plugins

from skimage._doctest_adapters import adapt_doctests  # noqa: E402

adapt_doctests(globals())
