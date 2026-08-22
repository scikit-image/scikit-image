"""Image Processing for Python (EXPERIMENTAL API version 2)."""

import sys
import importlib
import warnings

import _skimage2


# Will simulate the namespace of `_skimage2` as close as possible.
# This does not seem to work for `help(skimage2)` though and won't fool
# inspection tools, that look for actual subdirectories.
__all__ = _skimage2.__all__
__dir__ = _skimage2.__dir__
__getattr__ = _skimage2.__getattr__

# Register public submodules in sys.modules so they can be imported directly
# (e.g., `from skimage2.filters import gaussian`)
for _submod in __all__:
    if _submod not in {"__version__", "ExperimentalAPIWarning"}:
        sys.modules[f"skimage2.{_submod}"] = importlib.import_module(
            f"_skimage2.{_submod}"
        )


warnings.warn(
    "Importing from the `skimage2` namespace is experimental. "
    "Its API is under development and considered unstable!",
    category=_skimage2.ExperimentalAPIWarning,
    stacklevel=2,
)
