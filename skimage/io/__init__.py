"""Utilities to read and write images in various formats."""

import warnings

from .manage_plugins import *
from .manage_plugins import _hide_plugin_deprecation_warnings
from .sift import *
from .collection import *

from ._io import *
from ._image_stack import *


with _hide_plugin_deprecation_warnings():
    reset_plugins()


__all__ = [
    "call_plugin",
    "concatenate_images",
    "find_available_plugins",
    "imread",
    "imread_collection",
    "imread_collection_wrapper",
    "imsave",
    "imshow",
    "imshow_collection",
    "load_sift",
    "load_surf",
    "plugin_info",
    "plugin_order",
    "pop",
    "push",
    "reset_plugins",
    "show",
    "use_plugin",
    "ImageCollection",
    "MultiImage",
]


def __getattr__(name):
    if name == "available_plugins":
        warnings.warn(
            "`available_plugins` is deprecated since version 0.25 and will "
            "be removed in version 0.27. Use imageio or a similar package "
            "instead.",
            category=FutureWarning,
            stacklevel=2,
        )
        return globals()["_available_plugins"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
