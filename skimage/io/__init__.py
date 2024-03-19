"""Utilities to read and write images in various formats.
"""

import warnings
import functools
from contextlib import contextmanager

from .._shared.utils import deprecate_func


@contextmanager
def _hide_repeated_plugin_deprecation_warnings():
    """Ignore warnings related to plugin infrastructure deprecation."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message=".*Use imageio or a similar package instead.*",
            category=FutureWarning,
            module="skimage",
        )
        yield


def _deprecate_plugin_function(func):
    """Mark a function of the plugin infrastructure as deprecated.

    In addition to emitting the appropriate FutureWarning, this also supresses
    identical warnings that might be caused when this function calls other
    functions from the deprecated plugin infrastructure.
    """

    @deprecate_func(
        deprecated_version="0.23",
        removed_version="0.25",
        hint="Use imageio or a similar package instead.",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _hide_repeated_plugin_deprecation_warnings():
            return func(*args, **kwargs)

    return wrapper


from .manage_plugins import *
from .sift import *
from .collection import *

from ._io import *
from ._image_stack import *


__all__ = [
    'use_plugin',
    'call_plugin',
    'plugin_info',
    'plugin_order',
    'reset_plugins',
    'find_available_plugins',
    'available_plugins',
    'load_sift',
    'load_surf',
    'MultiImage',
    'ImageCollection',
    'concatenate_images',
    'imread_collection_wrapper',
    'imread',
    'imsave',
    'imshow',
    'show',
    'imread_collection',
    'imshow_collection',
    'image_stack',
    'push',
    'pop',
]


def __getattr__(name):
    if name == "available_plugins":
        warnings.warn(
            "`available_plugins` is deprecated since version 0.23 and will "
            "be removed in version 0.25. Use imageio or a similar package "
            "instead.",
            category=FutureWarning,
            stacklevel=2,
        )
        return globals()["_available_plugins"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
