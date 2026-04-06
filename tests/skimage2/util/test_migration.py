"""Test migration module"""

from textwrap import indent

import numpy as np

from _skimage2.util.migration import Skimage2Migration, ski2_migration_dec

import pytest

example_doc = """\
Replace all calls to `%(ski1qual)s` with `%(ski2qual)s`.

<!--- cond-start: warning -->
Only in warning
<!--- cond-end -->

<!--- cond-start: doc -->
Only in doc
## Examples

>>> import skimage as ski1
>>> import _skimage2 as ski2
>>> res1 = ski1.somemod.somefunc(10, 11)
>>> res2 = ski2.somemod.somefunc(10, 11)
>>> assert res1 == res2

Some background on the changes.
<!--- cond-end -->
"""


def test_parsing():
    migration_dec = Skimage2Migration()

    warn_msg, doc = migration_dec._parse_migration_doc(example_doc)
    assert 'Only in warning' in warn_msg
    assert 'Only in warning' not in doc
    assert 'Only in doc' in doc
    assert 'Only in doc' not in warn_msg


def func(a, b):
    return a * b


func_ski1qual = f'{func.__module__}.{func.__qualname__}'
example_filled = example_doc % dict(ski1qual=func_ski1qual,
                                    ski2qual=func_ski1qual)
warn_msg, doc = Skimage2Migration()._parse_migration_doc(example_filled)


def test_decoration_interpolation():
    migration_dec = Skimage2Migration(warn=True)
    dfunc = migration_dec(example_doc)(func)

    docs = migration_dec.migration_docs
    assert docs == {func_ski1qual: doc}

    from skimage.util import PendingSkimage2Change

    with pytest.warns(PendingSkimage2Change, match=warn_msg):
        assert dfunc(2, 4) == 8

    # Specify canonical location.
    migration_dec(example_doc, 'skimage.bar.baz')(func)
    assert docs['skimage.bar.baz'].startswith(
        'Replace all calls to `skimage.bar.baz` with `skimage2.bar.baz`.')
    # And skimage2 location.
    migration_dec(example_doc,
                  'skimage.bar.boo',
                  'skimage2.bun.biz')(func)
    assert docs['skimage.bar.boo'].startswith(
        'Replace all calls to `skimage.bar.boo` with `skimage2.bun.biz`.')


def test_dedent():
    # Test text dedented.
    migration_dec = Skimage2Migration(warn=True)
    dfunc = migration_dec(indent(example_doc, '    '))(func)

    from skimage.util import PendingSkimage2Change

    # Warning and doc nevertheless stays the same.
    assert migration_dec.migration_docs == {func_ski1qual: doc}
    with pytest.warns(PendingSkimage2Change, match=warn_msg):
        assert dfunc(2, 4) == 8


def test_peak_local_max(monkeypatch):

    from skimage.feature import peak_local_max
    from skimage.util import PendingSkimage2Change

    assert 'skimage.feature.peak_local_max' in ski2_migration_dec.migration_docs

    img = np.zeros((10, 10))

    monkeypatch.setattr(ski2_migration_dec, 'warn', True)
    with pytest.warns(PendingSkimage2Change,
                      match=('`skimage.feature.peak_local_max` '
                             'is deprecated in favor of')):
        peak_local_max(img)
