"""

Testing utilities.

"""

from _skimage2._shared.testing import (
    SKIP_RE as SKIP_RE,
    arch32 as arch32,
    assert_greater as assert_greater,
    assert_less as assert_less,
    assert_stacklevel as assert_stacklevel,
    color_check as color_check,
    doctest_skip_parser as doctest_skip_parser,
    fetch as fetch,
    fixture as fixture,
    mono_check as mono_check,
    parametrize as parametrize,
    raises as raises,
    roundtrip as roundtrip,
    run_in_parallel as run_in_parallel,
    skipif as skipif,
    xfail as xfail,
)  # noqa: F401

__all__ = [
    'SKIP_RE',
    'arch32',
    'assert_greater',
    'assert_less',
    'assert_stacklevel',
    'color_check',
    'doctest_skip_parser',
    'fetch',
    'fixture',
    'mono_check',
    'parametrize',
    'raises',
    'roundtrip',
    'run_in_parallel',
    'skipif',
    'xfail',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
