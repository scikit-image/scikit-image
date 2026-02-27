"""Expose API that is implemented outside `skimage.morphology`.

Lazy loader doesn't support outside-module imports [1]_.

.. [1] https://github.com/scientific-python/lazy-loader/issues/52
"""

from ..measure._label import label
