"""skimage2 namespace"""

from skimage import __version__

import warnings

import lazy_loader as _lazy


class ExperimentalAPIWarning(UserWarning):
    pass


warnings.warn(
    "Importing from the `skimage2` namespace is experimental. "
    "It's API is under development and considered unstable!",
    category=ExperimentalAPIWarning,
    stacklevel=2,
)

__getattr__, _, __all__ = _lazy.attach_stub(__name__, __file__)
