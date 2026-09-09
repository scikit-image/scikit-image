# Explicitly setting `__all__` is necessary for type inference engines
# to know which symbols are exported. See
# https://peps.python.org/pep-0484/#stub-files

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

from .manage_plugins import (
    use_plugin,
    call_plugin,
    plugin_info,
    plugin_order,
    reset_plugins,
    find_available_plugins,
)
from .sift import load_sift, load_surf
from .collection import (
    MultiImage,
    ImageCollection,
    concatenate_images,
    imread_collection_wrapper,
)
from ._io import imread, imsave, imread_collection
from ._image_stack import image_stack, push, pop
