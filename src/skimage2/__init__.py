"""Image Processing for Python (EXPERIMENTAL API version 2)."""

import importlib
import warnings

from _skimage2 import ExperimentalAPIWarning


__all__ = [
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
    "segmentation",
    "transform",
    "util",
]


def __getattr__(name):
    if name in __all__:
        submod = importlib.import_module(f"_skimage2.{name}")
        globals()[name] = submod
        return submod
    raise AttributeError(f"module 'skimage2' has no attribute '{name}'")


warnings.warn(
    "Importing from the `skimage2` namespace is experimental. "
    "Its API is under development and considered unstable!",
    ExperimentalAPIWarning,
)
