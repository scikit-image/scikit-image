"""Example images and datasets.

A curated set of general purpose and scientific images used in tests, examples,
and documentation.

No image files ship inside this package. The most frequently used ones come
from the ``scikit-image-data`` package; install ``scikit-image[data]`` to be
sure they are present and need no download. Every other dataset is downloaded
on demand and cached in ``data_dir``, so ``data_dir`` holds only what has been
downloaded so far. To make every dataset available offline, use
:func:`download_all`, which also copies the ``scikit-image-data`` images into
``data_dir``.

"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
