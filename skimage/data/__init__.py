"""
Images and datasets for tests and examples.

Makes a curated set of general purpose and scientific images and datasets easily
available for tests, examples and documentation. Newer datasets are no longer included
in scikit-image directly but are downloaded if accessed. To make them available offline,
use :func:`download_all`.

"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
