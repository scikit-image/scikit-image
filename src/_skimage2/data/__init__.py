"""Example images and datasets.

A curated set of general-purpose and scientific images used in tests, examples,
and documentation. Also includes functionality to generate synthetic data.
"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
