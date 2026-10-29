"""Image Processing for Python (EXPERIMENTAL API version 2)."""

import warnings

import lazy_loader as _lazy

from _skimage2 import __version__
from _skimage2._shared._warnings import ExperimentalAPIWarning


# Each submodule is a shim package that forwards to its `_skimage2` counterpart.
__getattr__, __dir__, __all__ = _lazy.attach(
    __name__,
    submodules=[
        "color",
        "data",
        "draw",
        "exposure",
        "feature",
        "filters",
        "graph",
        "io",
        "measure",
        "metrics",
        "morphology",
        "registration",
        "restoration",
        "segmentation",
        "transform",
        "util",
    ],
)
__all__ += ["__version__", "ExperimentalAPIWarning"]


warnings.warn(
    "Importing from the `skimage2` namespace is experimental. "
    "Its API is under development and considered unstable!",
    category=ExperimentalAPIWarning,
    stacklevel=2,
)
