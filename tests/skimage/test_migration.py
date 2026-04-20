"""Test migration module"""

import numpy as np

from skimage._migration import Skimage2Migration, ski2_migration_decorator

import pytest

EXAMPLE_INPUT = """\
Replace all calls to ``%(qname_old)s`` with ``%(qname_new)s``.

.. code-block:: python
    :linenos:
    :emphasize-lines: 1

    a = 1
    a

<!--- cond-start: warning -->
Only in warning

.. code-block:: python
    print('foo')

<!--- cond-end -->
  .. code-block:: python

    b = 2
    b

<!--- cond-start: doc -->
Only in doc

Examples
--------

>>> import skimage as ski1
>>> import _skimage2 as ski2
>>> res1 = ski1.somemod.somefunc(10, 11)
>>> res2 = ski2.somemod.somefunc(10, 11)
>>> assert res1 == res2

Some background on the changes.
<!--- cond-end -->
"""

EXAMPLE_WARN = """\
Replace all calls to `%(qname_old)s` with `%(qname_new)s`.

    a = 1
    a

Only in warning

    print('foo')

    b = 2
    b

See %(migration_url)s#%(qname_old_anchor)s
""".strip()

EXAMPLE_DOC = """\
Replace all calls to ``%(qname_old)s`` with ``%(qname_new)s``.

.. code-block:: python
    :linenos:
    :emphasize-lines: 1

    a = 1
    a

  .. code-block:: python

    b = 2
    b

Only in doc

Examples
--------

>>> import skimage as ski1
>>> import _skimage2 as ski2
>>> res1 = ski1.somemod.somefunc(10, 11)
>>> res2 = ski2.somemod.somefunc(10, 11)
>>> assert res1 == res2

Some background on the changes.
""".strip()


MIGRATION_URL = 'https://some.site/doc/migration.html'


def func(a, b):
    return a * b


@pytest.fixture
def md_dfunc():
    migration_dec = Skimage2Migration(MIGRATION_URL)
    return migration_dec, migration_dec(EXAMPLE_INPUT)(func)


def test_parsing(md_dfunc):
    migration_dec, _ = md_dfunc
    warn_msg, doc = migration_dec._parse_migration_doc(EXAMPLE_INPUT)
    assert warn_msg == EXAMPLE_WARN
    assert doc == EXAMPLE_DOC


_func_qname_old = f'{func.__module__}.{func.__qualname__}'
_anchor = _func_qname_old.replace('.', '-').replace('_', '-')
warn_msg, doc = Skimage2Migration(MIGRATION_URL)._filled_docs(
    EXAMPLE_INPUT,
    dict(
        qname_old=_func_qname_old,
        qname_new=_func_qname_old,
        migration_url=MIGRATION_URL,
        qname_old_anchor=_anchor,
    ),
)


def test_decoration_interpolation(md_dfunc):
    migration_dec, dfunc = md_dfunc

    docs = migration_dec.migration_docs
    assert docs == {_func_qname_old: doc}

    from skimage.util import PendingSkimage2Change

    with pytest.warns(PendingSkimage2Change) as record:
        assert dfunc(2, 4) == 8

    assert len(record) == 1
    assert record[0].message.args[0] == warn_msg

    # Specify canonical location.
    migration_dec(EXAMPLE_INPUT, 'skimage.bar.baz')(func)
    assert docs['skimage.bar.baz'].startswith(
        'Replace all calls to ``skimage.bar.baz`` with ``skimage2.bar.baz``.'
    )
    # And skimage2 location.
    migration_dec(EXAMPLE_INPUT, 'skimage.bar.boo', 'skimage2.bun.biz')(func)
    assert docs['skimage.bar.boo'].startswith(
        'Replace all calls to ``skimage.bar.boo`` with ``skimage2.bun.biz``.'
    )


def test_dedent(md_dfunc):
    # Test text dedented.
    migration_dec, dfunc = md_dfunc

    from skimage.util import PendingSkimage2Change

    # Warning and doc nevertheless stays the same.
    assert migration_dec.migration_docs == {_func_qname_old: doc}
    with pytest.warns(PendingSkimage2Change) as record:
        assert dfunc(2, 4) == 8

    assert len(record) == 1
    assert record[0].message.args[0] == warn_msg


def test_peak_local_max():
    from skimage.feature import peak_local_max
    from skimage.util import PendingSkimage2Change

    assert 'skimage.feature.peak_local_max' in ski2_migration_decorator.migration_docs

    img = np.zeros((10, 10))

    with pytest.warns(
        PendingSkimage2Change,
        match=('`skimage.feature.peak_local_max` ' 'is deprecated in favor of'),
    ):
        peak_local_max(img)


def test_comment_check(md_dfunc):
    migration_dec, _ = md_dfunc
    doc = EXAMPLE_INPUT + '\n\nA <!-- marker'
    with pytest.raises(
        ValueError, match=r"Remaining <!-- marker in warning of `foo\.bar`;"
    ):
        migration_dec._parse_migration_doc(doc, 'foo.bar')
