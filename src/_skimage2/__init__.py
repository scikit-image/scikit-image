"""Image Processing for Python (EXPERIMENTAL API version 2).

This internal package is where the actual implementation of the API v2 lives
and where it can be imported from without triggering a `ExperimentalAPIWarning`.
Once the API is declared *stable*, its implementation will move into the
`skimage2` namespace.
"""

import lazy_loader as _lazy


class ExperimentalAPIWarning(UserWarning):
    """Mark unstable API that's intentionally not published (yet)."""


_stub_getattr, _, _stub_all = _lazy.attach_stub(__name__, __file__)


# `lazy_loader.attach_stub` is ignoring the stub's `__all__`, so patch it here
__all__ = _stub_all + ["__version__", "ExperimentalAPIWarning"]


def __dir__():
    return __all__.copy()


def __getattr__(name):
    if name == "__version__":
        # TODO Undo inlined imports once ported
        from skimage import __version__

        return __version__

    return _stub_getattr(name)
