from _skimage2._shared.utils import (
    deprecate_func as deprecate_func,
    get_bound_method_class as get_bound_method_class,
    all_warnings as all_warnings,
    safe_as_int as safe_as_int,
    check_shape_equality as check_shape_equality,
    check_nD as check_nD,
    warn as warn,
    reshape_nd as reshape_nd,
    identity as identity,
    slice_at_axis as slice_at_axis,
    deprecate_parameter as deprecate_parameter,
    DEPRECATED as DEPRECATED,
)  # noqa: F401

__all__ = [
    'deprecate_func',
    'get_bound_method_class',
    'all_warnings',
    'safe_as_int',
    'check_shape_equality',
    'check_nD',
    'warn',
    'reshape_nd',
    'identity',
    'slice_at_axis',
    'deprecate_parameter',
    'DEPRECATED',
]

from skimage._doctest_adapters import adapt_doctests  # noqa: E402

adapt_doctests(globals())
