from _skimage2.color.colorlabel import (
    color_dict as color_dict,
    label2rgb as label2rgb,
    DEFAULT_COLORS as DEFAULT_COLORS,
)  # noqa: F401

__all__ = [
    'color_dict',
    'label2rgb',
    'DEFAULT_COLORS',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
