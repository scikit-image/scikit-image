"""Rank filters, e.g. median, entropy, local Otsu, etc."""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)

__3Dfilters = [
    'autolevel',
    'equalize',
    'gradient',
    'majority',
    'maximum',
    'mean',
    'geometric_mean',
    'subtract_mean',
    'median',
    'minimum',
    'modal',
    'enhance_contrast',
    'pop',
    'sum',
    'threshold',
    'noise_filter',
    'entropy',
    'otsu',
]
