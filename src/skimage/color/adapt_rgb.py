from _skimage2.color.adapt_rgb import (
    adapt_rgb as adapt_rgb,
    hsv_value as hsv_value,
    each_channel as each_channel,
)  # noqa: F401

__all__ = [
    'adapt_rgb',
    'hsv_value',
    'each_channel',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
