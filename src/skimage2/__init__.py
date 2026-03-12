"""skimage2 namespace"""

import warnings

import lazy_loader as _lazy


class ExperimentalAPIWarning(UserWarning):
    """Mark unstable API that's intentionally not published (yet)."""


warnings.warn(
    "Importing from the `skimage2` namespace is experimental. "
    "Its API is under development and considered unstable!",
    category=ExperimentalAPIWarning,
    stacklevel=2,
)


_stub_getattr, _, __all__ = _lazy.attach_stub(__name__, __file__)


def __getattr__(name):
    if name == "__version__":
        # TODO Undo inlined imports once ported
        from skimage import __version__

        return __version__

    return _stub_getattr(name)
