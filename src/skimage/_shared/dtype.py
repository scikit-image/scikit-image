from _skimage2._shared.dtype import (
    bool_types as bool_types,
    complex_dtypes as complex_dtypes,
    complex_types as complex_types,
    floating_dtypes as floating_dtypes,
    floating_types as floating_types,
    inexact_dtypes as inexact_dtypes,
    inexact_types as inexact_types,
    integer_dtypes as integer_dtypes,
    integer_types as integer_types,
    numeric_dtype_min_max as numeric_dtype_min_max,
    numeric_dtypes as numeric_dtypes,
    numeric_types as numeric_types,
    signed_integer_dtypes as signed_integer_dtypes,
    signed_integer_types as signed_integer_types,
    unsigned_integer_dtypes as unsigned_integer_dtypes,
)  # noqa: F401

__all__ = [
    'bool_types',
    'complex_dtypes',
    'complex_types',
    'floating_dtypes',
    'floating_types',
    'inexact_dtypes',
    'inexact_types',
    'integer_dtypes',
    'integer_types',
    'numeric_dtype_min_max',
    'numeric_dtypes',
    'numeric_types',
    'signed_integer_dtypes',
    'signed_integer_types',
    'unsigned_integer_dtypes',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
