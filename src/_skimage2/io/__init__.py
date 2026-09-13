"""Reading and saving of images and videos."""

import warnings

import lazy_loader as _lazy

from .manage_plugins import (
    _available_plugins,
    _hide_plugin_deprecation_warnings,
    reset_plugins,
)

_lazy_getattr, __dir__, _ = _lazy.attach_stub(__name__, __file__)

# Don't use the `__all__` returned by `attach_stub`; the plugin management
# functions are importable but not advertised as part of the public API.
__all__ = [
    "concatenate_images",
    "imread",
    "imread_collection",
    "imread_collection_wrapper",
    "imsave",
    "load_sift",
    "load_surf",
    "pop",
    "push",
    "ImageCollection",
    "MultiImage",
]

with _hide_plugin_deprecation_warnings():
    reset_plugins()


def __getattr__(name):
    if name == "available_plugins":
        warnings.warn(
            "`available_plugins` is deprecated since version 0.25 and will "
            "be removed in version 0.27. Instead, use `imageio` or other "
            "I/O packages directly.",
            category=FutureWarning,
            stacklevel=2,
        )
        return _available_plugins
    return _lazy_getattr(name)
