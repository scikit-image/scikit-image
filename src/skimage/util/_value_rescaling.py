from _skimage2.util._value_rescaling import (
    rescale_legacy as rescale_legacy,
    rescale_minmax as rescale_minmax,
)  # noqa: F401

__all__ = [
    'rescale_legacy',
    'rescale_minmax',
]

from _skimage2.util._value_rescaling import _prescale_value_range  # noqa: F401
