from _skimage2.util.dtype import (
    img_as_float32 as img_as_float32,
    img_as_float64 as img_as_float64,
    img_as_float as img_as_float,
    img_as_int as img_as_int,
    img_as_uint as img_as_uint,
    img_as_ubyte as img_as_ubyte,
    img_as_bool as img_as_bool,
    dtype_limits as dtype_limits,
)  # noqa: F401

__all__ = [
    'img_as_float32',
    'img_as_float64',
    'img_as_float',
    'img_as_int',
    'img_as_uint',
    'img_as_ubyte',
    'img_as_bool',
    'dtype_limits',
]

from _skimage2.util.dtype import _convert, _integer_types, convert, dtype_range  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
