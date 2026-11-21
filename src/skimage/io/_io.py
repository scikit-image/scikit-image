import warnings

from _skimage2.io._io import (
    imread as imread,
    imsave as imsave,
    imread_collection as imread_collection,
)  # noqa: F401

from _skimage2._shared.utils import deprecate_parameter, DEPRECATED
from _skimage2.io._io import _remove_plugin_param_template
from _skimage2.io.manage_plugins import call_plugin, _hide_plugin_deprecation_warnings

# Import registers the shim ImageCollection, for `as_shim`.
from . import collection  # noqa: F401

__all__ = [
    'imread',
    'imsave',
    'imread_collection',
]

from skimage._doctest_adapters import adapt_doctests, as_shim

adapt_doctests(globals(), skip_names=('DEPRECATED', 'deprecate_parameter'))

_ski2_imread_collection_doc = imread_collection.__doc__


# Same as the `_skimage2` function, but returns the `skimage` ImageCollection.
# The body is not delegated, so that warnings keep their stack level.
@deprecate_parameter(
    "plugin",
    start_version="0.25",
    stop_version="0.27",
    template=_remove_plugin_param_template,
)
def imread_collection(
    load_pattern, conserve_memory=True, plugin=DEPRECATED, **plugin_args
):
    if plugin is DEPRECATED:
        plugin = None
    if plugin_args:
        msg = (
            "The I/O plugin infrastructure is deprecated since "
            "version 0.25 and will be removed in 0.27 (or later). To avoid "
            "this warning, please do not pass additional keyword arguments "
            "for plugins (`**plugin_args`). Instead, use `imageio` or other "
            "I/O packages directly. See also `imread_collection`."
        )
        warnings.warn(msg, category=FutureWarning, stacklevel=3)
    with _hide_plugin_deprecation_warnings():
        collection = call_plugin(
            'imread_collection',
            load_pattern,
            conserve_memory,
            plugin=plugin,
            **plugin_args,
        )
    return as_shim(collection)


imread_collection.__doc__ = _ski2_imread_collection_doc
