"""Test migration module"""

from _skimage2.util.migration import Skimage2Migration

import pytest

example_doc = """\
## Summary

Replace all calls to `%(ski1qual)s` with `%(ski2qual)s`.

## Examples

>>> import skimage as ski1
>>> import _skimage2 as ski2
>>> res1 = ski1.somemod.somefunc(10, 11)
>>> res2 = ski2.somemod.somefunc(10, 11)
>>> assert res1 == res2

## Background

Some background on the changes.
"""


def test_parsing():
    migration_dec = Skimage2Migration()

    parts = migration_dec._parse_migration_doc(example_doc)
    full_parts = {'Summary', 'Examples', 'Background'}
    assert set(parts) == full_parts
    low_summary = example_doc
    for heading in full_parts:
        low_summary = low_summary.replace(heading, heading.lower())
        assert set(migration_dec._parse_migration_doc(low_summary)) == full_parts
    bad_background = low_summary.replace('background', 'backgrounds')
    assert set(migration_dec._parse_migration_doc(bad_background)) == {
        'Summary',
        'Examples',
    }
    bad_summary = example_doc.replace('Summary', 'No summary')
    with pytest.raises(ValueError, match='Migration message should contain a summary'):
        migration_dec._parse_migration_doc(bad_summary)


def func(a, b):
    return a * b


_func_ski1qual = f'{func.__module__}.{func.__qualname__}'
example_filled = example_doc % dict(ski1qual=_func_ski1qual, ski2qual=_func_ski1qual)


def test_decoration_interpolation():
    migration_dec = Skimage2Migration()
    parsed_eg_doc = migration_dec._parse_migration_doc(example_filled)
    dfunc = migration_dec(example_doc)(func)

    assert migration_dec.migration_messages == {
        f'{func.__module__}.{func.__qualname__}': parsed_eg_doc
    }

    from skimage.util import PendingSkimage2Change

    with pytest.warns(PendingSkimage2Change, match=parsed_eg_doc['Summary']):
        assert dfunc(2, 4) == 8
