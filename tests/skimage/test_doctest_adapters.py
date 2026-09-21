import sys
import types

import numpy as np

from skimage._doctest_adapters import (
    _adapt_doctest_doc,
    _adapt_obj_doctest,
    adapt_doctests,
)

import pytest


def skip_if_pyopt2(func):
    return pytest.mark.skipif(
        sys.flags.optimize >= 2,
        reason='PYTHONOPTIMIZE=2 strips docstrings',
    )(func)


def test_adapt_import_lines():
    doc = """\
Examples
--------
>>> from _skimage2 import data
>>> from _skimage2.transform import resize
>>> import _skimage2 as ski2
>>> ski2.data
"""
    adapted = _adapt_doctest_doc(doc)
    expected = """\
Examples
--------
>>> from skimage import data
>>> from skimage.transform import resize
>>> import skimage as ski
>>> ski.data
"""
    assert adapted == expected
    doc = """\
    Examples
    --------
    >>> import _skimage2 as ski2
    >>> image_shape = (10, 10)
    >>> polygon = np.array([[1, 1], [2, 7], [8, 4]])
    >>> mask = ski2.draw.polygon2mask(image_shape, polygon)
    """
    adapted = _adapt_doctest_doc(doc)
    expected = """\
    Examples
    --------
    >>> import skimage as ski
    >>> image_shape = (10, 10)
    >>> polygon = np.array([[1, 1], [2, 7], [8, 4]])
    >>> mask = ski.draw.polygon2mask(image_shape, polygon)
    """
    assert adapted == expected


def test_adapt_expected_output():
    doc = """\
>>> ski2.util.lookfor('regular_grid')
Search results for 'regular_grid'
---------------------------------
_skimage2.util.regular_grid
    Find `n_points` regularly spaced along `ar_shape`.
"""
    adapted = _adapt_doctest_doc(doc)
    assert adapted == (
        doc.replace('_skimage2.util', 'skimage.util').replace('ski2.util', 'ski.util')
    )


@skip_if_pyopt2
def test_prose_outside_doctests_unchanged():
    doc = """\
See :func:`skimage2.util.img_as_float` for details.

Examples
--------
>>> from _skimage2 import data
"""
    adapted = _adapt_doctest_doc(doc)
    assert adapted == doc.replace('from _skimage2', 'from skimage')


@skip_if_pyopt2
def test_adapt_obj_doctest_sets_doc():
    def func():
        """Example

        >>> from _skimage2 import data
        """
        return 1

    bound = _adapt_obj_doctest(func, shim_module='skimage.tests.example')
    assert bound.__doc__ == func.__doc__.replace('_skimage2', 'skimage')
    assert bound.__module__ == 'skimage.tests.example'
    assert bound() == 1


@skip_if_pyopt2
def test_adapt_obj_doctest_class_uses_proxy_without_mutating_impl():
    class Impl:
        """Example

        >>> from _skimage2 import data
        """

    assert Impl.__module__ == __name__
    in_doc = Impl.__doc__
    assert '_skimage2' in in_doc
    bound = _adapt_obj_doctest(Impl, shim_module='skimage.tests.example')
    assert bound is not Impl
    assert issubclass(bound, Impl)
    assert bound.__doc__ == in_doc.replace('_skimage2', 'skimage')
    assert bound.__module__ == 'skimage.tests.example'


@skip_if_pyopt2
def test_adapt_doctests():
    def one():
        """>>> from _skimage2 import data"""

    def two():
        """>>> import _skimage2 as ski2"""

    mod = types.ModuleType('fake_module')
    mod.one = one
    mod.two = two
    ns = mod.__dict__
    adapt_doctests(ns)
    assert ns['one'].__doc__ == '>>> from skimage import data'
    assert ns['two'].__doc__ == '>>> import skimage as ski'


@skip_if_pyopt2
def test_adapt_doctests_defaults_to_caller_globals():
    def shim_func():
        """>>> from _skimage2 import data"""

    caller_ns = {
        'shim_func': shim_func,
        '__name__': 'skimage.tests.example_shim',
        'adapt_doctests': adapt_doctests,
    }
    exec('adapt_doctests(globals())', caller_ns)
    assert caller_ns['shim_func'].__doc__ == '>>> from skimage import data'


@skip_if_pyopt2
def test_adapt_doctests_copies_doctest_requires():
    impl = types.ModuleType('_skimage2.tests.example_impl')
    impl.__doctest_requires__ = {'func': ['matplotlib']}

    def func():
        """>>> from _skimage2 import data"""

    func.__module__ = impl.__name__
    impl.func = func

    ns = {
        'func': func,
        '__name__': 'skimage.tests.example_shim',
    }

    try:
        sys.modules[impl.__name__] = impl
        adapt_doctests(ns)
        assert ns['__doctest_requires__'] == {'func': ['matplotlib']}
    finally:
        sys.modules.pop(impl.__name__, None)


def test_adapt_doctests_injects_np():
    ns = {'__name__': 'skimage.tests.example_shim'}
    assert 'np' not in ns
    adapt_doctests(ns)
    assert ns['np'] is np


def test_shim_draw_propagates_doctest_requires():
    import skimage.draw.draw as ski1_draw_mod
    import _skimage2.draw.draw as ski2_draw_mod

    dt_req1 = getattr(ski1_draw_mod, '__doctest_requires__')
    dt_req2 = getattr(ski2_draw_mod, '__doctest_requires__')
    for func_name in ('polygon_perimeter', 'rectangle_perimeter'):
        assert dt_req1[func_name] == dt_req2[func_name]


def test_adapt_preserves_output_block():
    doc = """\
    >>> A = 1
    >>> A
    array([1])
      More output
    >>> B = 2
    array([2])
     Yet more
    """
    assert _adapt_doctest_doc(doc) == doc


# --------------------------------------------------------------------------
# Method-level docstring adaptation
# --------------------------------------------------------------------------


@skip_if_pyopt2
def test_adapt_obj_doctest_adapts_method_docstrings():
    class Impl:
        """Example class.

        >>> from _skimage2 import data
        """

        def process(self):
            """Process the image.

            >>> from _skimage2.transform import resize
            """
            return 42

        def untouched_method(self):
            """No references here."""
            return 0

    proxy = _adapt_obj_doctest(Impl, shim_module='skimage.tests.example')

    # Class-level doc adapted
    assert proxy.__doc__ == Impl.__doc__.replace('_skimage2', 'skimage')

    # Method doc adapted
    assert proxy.process.__doc__ == Impl.process.__doc__.replace('_skimage2', 'skimage')

    # Method without _skimage2 references is NOT overridden — it comes from
    # the parent, so its __doc__ is the original.
    assert proxy.untouched_method.__doc__ == Impl.untouched_method.__doc__

    # Behavior unchanged
    assert proxy().process() == 42
    assert Impl().process() == 42

    # Original class and methods are unmodified
    assert '_skimage2' in Impl.__doc__
    assert '_skimage2' in Impl.process.__doc__
    assert Impl.__module__ == __name__


@skip_if_pyopt2
def test_adapt_obj_doctest_adapts_classmethod_docstrings():
    class Impl:
        @classmethod
        def create(cls):
            """Create an instance.

            >>> import _skimage2 as ski2
            """
            return cls()

    proxy = _adapt_obj_doctest(Impl, shim_module='skimage.tests.example')

    proxy_create = vars(proxy).get('create')
    assert proxy_create is not None, 'classmethod should be on proxy, not inherited'
    assert proxy_create.__func__.__doc__ == Impl.create.__doc__.replace(
        '_skimage2', 'skimage'
    ).replace('ski2', 'ski')

    # Original unmodified
    assert '_skimage2' in Impl.create.__func__.__doc__


@skip_if_pyopt2
def test_adapt_obj_doctest_adapts_staticmethod_docstrings():
    class Impl:
        @staticmethod
        def helper(x):
            """Helpful helper.

            >>> from _skimage2 import data
            """
            return x * 2

    proxy = _adapt_obj_doctest(Impl, shim_module='skimage.tests.example')

    proxy_helper = vars(proxy).get('helper')
    assert proxy_helper is not None, 'staticmethod should be on proxy, not inherited'
    assert proxy_helper.__func__.__doc__ == Impl.helper.__doc__.replace(
        '_skimage2', 'skimage'
    )

    # Original unmodified — staticmethod descriptor returns raw function
    assert '_skimage2' in Impl.helper.__doc__


@skip_if_pyopt2
def test_adapt_obj_doctest_class_method_without_skimage_ref_not_copied():
    """Methods without _skimage2 references stay inherited, not overridden."""

    class Impl:
        """Example class.

        >>> from _skimage2 import data
        """

        def plain(self):
            """No namespace references."""
            return 1

    proxy = _adapt_obj_doctest(Impl, shim_module='skimage.tests.example')

    # 'plain' should NOT appear in the proxy's own dict — it is inherited
    assert 'plain' not in vars(proxy)
    assert proxy.plain.__doc__ == 'No namespace references.'


@skip_if_pyopt2
def test_adapt_obj_doctest_adapts_init_docstring():
    class Impl:
        """Example class.

        >>> from _skimage2 import data
        """

        def __init__(self, x):
            """Initialize.

            >>> import _skimage2 as ski2
            """
            self.x = x

    proxy = _adapt_obj_doctest(Impl, shim_module='skimage.tests.example')

    # __init__ docstring adapted
    proxy_init = vars(proxy).get('__init__')
    assert proxy_init is not None
    assert proxy_init.__doc__ == Impl.__init__.__doc__.replace(
        '_skimage2', 'skimage'
    ).replace('ski2', 'ski')

    # Original unmodified
    assert '_skimage2' in Impl.__init__.__doc__
