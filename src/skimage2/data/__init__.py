"""Example images and datasets.

A curated set of general purpose and scientific images used in tests, examples,
and documentation.

Newer datasets are no longer included as part of the package, but are
downloaded on demand. To make data available offline, use :func:`download_all`."""

from _skimage2 import data as _impl

# `_skimage2.data` resolves its own attributes lazily; forward to it.
__getattr__ = _impl.__getattr__
__dir__ = _impl.__dir__
__all__ = _impl.__all__
