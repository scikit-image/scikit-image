"""Test migration module"""

from textwrap import indent

import numpy as np

from _skimage2.util.migration import Skimage2Migration, ski2_migration_dec

import pytest

EXAMPLE_INPUT = """\
Replace all calls to `%(ski1qual)s` with `%(ski2qual)s`.

<!--- cond-start: warning -->
Only in warning

```{python}
print('foo')
```
<!--- cond-end -->

  ```python
  a = 1
  a
  ```

<!--- cond-start: doc -->
Only in doc

## Examples

```{python}
import skimage as ski1
import _skimage2 as ski2
res1 = ski1.somemod.somefunc(10, 11)
res2 = ski2.somemod.somefunc(10, 11)
assert res1 == res2
```

Some background on the changes.
<!--- cond-end -->
"""

EXAMPLE_WARN = """\
Replace all calls to `%(ski1qual)s` with `%(ski2qual)s`.

Only in warning

  print('foo')

    a = 1
    a

See %(migration_url)s#%(ski1qual)s
""".strip()

EXAMPLE_DOC = """\
Replace all calls to `%(ski1qual)s` with `%(ski2qual)s`.


  ```python
  a = 1
  a
  ```

Only in doc

## Examples

```{python}
import skimage as ski1
import _skimage2 as ski2
res1 = ski1.somemod.somefunc(10, 11)
res2 = ski2.somemod.somefunc(10, 11)
assert res1 == res2
```

Some background on the changes.
""".strip()


MIGRATION_URL = 'https://some.site/doc/migration.html'


def test_parsing():
    migration_dec = Skimage2Migration(MIGRATION_URL)

    warn_msg, doc = migration_dec._parse_migration_doc(EXAMPLE_INPUT)
    assert warn_msg == EXAMPLE_WARN
    assert doc == EXAMPLE_DOC


def func(a, b):
    return a * b


_func_ski1qual = f'{func.__module__}.{func.__qualname__}'
warn_msg, doc = (Skimage2Migration(MIGRATION_URL)
                 ._filled_docs(EXAMPLE_INPUT,
                               dict(ski1qual=_func_ski1qual,
                                    ski2qual=_func_ski1qual,
                                    migration_url=MIGRATION_URL)))


def test_decoration_interpolation():
    migration_dec = Skimage2Migration(MIGRATION_URL)
    dfunc = migration_dec(EXAMPLE_INPUT)(func)

    docs = migration_dec.migration_docs
    assert docs == {_func_ski1qual: doc}

    from skimage.util import PendingSkimage2Change

    with pytest.warns(PendingSkimage2Change) as record:
        assert dfunc(2, 4) == 8

    assert len(record) == 1
    assert record[0].message.args[0] == warn_msg

    # Specify canonical location.
    migration_dec(EXAMPLE_INPUT, 'skimage.bar.baz')(func)
    assert docs['skimage.bar.baz'].startswith(
        'Replace all calls to `skimage.bar.baz` with `skimage2.bar.baz`.')
    # And skimage2 location.
    migration_dec(EXAMPLE_INPUT,
                  'skimage.bar.boo',
                  'skimage2.bun.biz')(func)
    assert docs['skimage.bar.boo'].startswith(
        'Replace all calls to `skimage.bar.boo` with `skimage2.bun.biz`.')


def test_dedent():
    # Test text dedented.
    migration_dec = Skimage2Migration(MIGRATION_URL)
    dfunc = migration_dec(indent(EXAMPLE_INPUT, '    '))(func)

    from skimage.util import PendingSkimage2Change

    # Warning and doc nevertheless stays the same.
    assert migration_dec.migration_docs == {_func_ski1qual: doc}
    with pytest.warns(PendingSkimage2Change) as record:
        assert dfunc(2, 4) == 8

    assert len(record) == 1
    assert record[0].message.args[0] == warn_msg

def test_peak_local_max():

    from skimage.feature import peak_local_max
    from skimage.util import PendingSkimage2Change

    assert 'skimage.feature.peak_local_max' in ski2_migration_dec.migration_docs

    img = np.zeros((10, 10))

    with pytest.warns(PendingSkimage2Change,
                      match=('`skimage.feature.peak_local_max` '
                             'is deprecated in favor of')):
        peak_local_max(img)
