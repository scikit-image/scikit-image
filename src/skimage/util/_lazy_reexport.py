"""Expose API that is implemented outside `skimage.util`.

Lazy loader doesn't support outside-module imports [1]_.

.. [1] https://github.com/scientific-python/lazy-loader/issues/52
"""

from .._shared.utils import FailedEstimationAccessError  # noqa: F401
