import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

from skimage._doctest_adapters import (
    adapt_doctest_doc,
    adapt_obj_doctest,
    adapt_doctests,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_adapt_import_lines():
    doc = """\
Examples
--------
>>> from _skimage2 import data
>>> from _skimage2.transform import resize
>>> import _skimage2 as ski2
>>> ski2.data
"""
    adapted = adapt_doctest_doc(doc)
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
    adapted = adapt_doctest_doc(doc)
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
    adapted = adapt_doctest_doc(doc)
    assert adapted == (
        doc.replace('_skimage2.util', 'skimage.util').replace('ski2.util', 'ski.util')
    )


def test_prose_outside_doctests_unchanged():
    doc = """\
See :func:`skimage2.util.img_as_float` for details.

Examples
--------
>>> from _skimage2 import data
"""
    adapted = adapt_doctest_doc(doc)
    assert adapted == doc.replace('from _skimage2', 'from skimage')


def test_adapt_obj_doctest_sets_doc():
    def func():
        """Example

        >>> from _skimage2 import data
        """
        return 1

    bound = adapt_obj_doctest(func, shim_module='skimage.tests.example')
    assert bound.__doc__ == func.__doc__.replace('_skimage2', 'skimage')
    assert bound.__module__ == 'skimage.tests.example'
    assert bound() == 1


def test_adapt_obj_doctest_class_uses_proxy_without_mutating_impl():
    class Impl:
        """Example

        >>> from _skimage2 import data
        """

    bound = adapt_obj_doctest(Impl, shim_module='skimage.tests.example')
    assert bound is not Impl
    assert issubclass(bound, Impl)
    assert 'from skimage import data' in bound.__doc__
    assert bound.__module__ == 'skimage.tests.example'
    assert '_skimage2' in (Impl.__doc__ or '')
    assert Impl.__module__ == __name__


def test_adapt_doctests():
    def one():
        """>>> from _skimage2 import data"""

    def two():
        """>>> import _skimage2 as ski"""

    mod = types.ModuleType('fake_module')
    mod.one = one
    mod.two = two
    ns = mod.__dict__
    adapt_doctests(ns)
    assert 'from skimage import data' in ns['one'].__doc__
    assert 'import skimage as ski' in ns['two'].__doc__


def test_adapt_doctests_defaults_to_caller_globals():
    def shim_func():
        """>>> from _skimage2 import data"""

    caller_ns = {
        'shim_func': shim_func,
        '__name__': 'skimage.tests.example_shim',
        'adapt_doctests': adapt_doctests,
    }
    exec('adapt_doctests(globals())', caller_ns)
    assert 'from skimage import data' in caller_ns['shim_func'].__doc__


def test_adapt_doctests_copies_doctest_requires():
    import sys
    import types

    impl = types.ModuleType('_skimage2.tests.example_impl')
    impl.__doctest_requires__ = {'func': ['matplotlib']}
    sys.modules[impl.__name__] = impl

    def func():
        """>>> from _skimage2 import data"""

    func.__module__ = impl.__name__
    impl.func = func

    ns = {
        'func': func,
        '__name__': 'skimage.tests.example_shim',
    }
    try:
        adapt_doctests(ns)
        assert ns['__doctest_requires__'] == {'func': ['matplotlib']}
    finally:
        sys.modules.pop(impl.__name__, None)


def test_adapt_doctests_injects_np():
    import numpy as np

    ns = {'__name__': 'skimage.tests.example_shim'}
    adapt_doctests(ns)
    assert ns['np'] is np


def test_shim_draw_propagates_doctest_requires():
    import _skimage2.draw.draw as ski2_draw_mod
    import skimage.draw.draw as draw_mod

    assert ski2_draw_mod.__doctest_requires__ == draw_mod.__doctest_requires__


@pytest.mark.skipif(
    importlib.util.find_spec('matplotlib') is not None,
    reason='requires matplotlib to be absent',
)
def test_shim_draw_skips_matplotlib_doctests():
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '--doctest-plus',
            '--pyargs',
            'skimage.draw.draw',
            '-k',
            'rectangle_perimeter or polygon_perimeter',
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'skipped' in result.stdout.lower()


def test_adapt_preserves_array_output_block():
    doc = """\
>>> A = 1
>>> A
array([1])
>>> B = 2
"""
    adapted = adapt_doctest_doc(doc)
    assert 'array([1])' in adapted


# --------------------------------------------------------------------------
# Method-level docstring adaptation
# --------------------------------------------------------------------------


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

    proxy = adapt_obj_doctest(Impl, shim_module='skimage.tests.example')

    # Class-level doc adapted
    assert 'from skimage import data' in proxy.__doc__

    # Method doc adapted
    assert 'from skimage.transform import resize' in proxy.process.__doc__

    # Method without _skimage2 references is NOT overridden — it comes from
    # the parent, so its __doc__ is the original.
    assert proxy.untouched_method.__doc__ == Impl.untouched_method.__doc__

    # Behavior unchanged
    assert proxy.process(None) == 42

    # Original class and methods are unmodified
    assert '_skimage2' in Impl.__doc__
    assert '_skimage2' in Impl.process.__doc__
    assert Impl.__module__ == __name__


def test_adapt_obj_doctest_adapts_classmethod_docstrings():
    class Impl:
        @classmethod
        def create(cls):
            """Create an instance.

            >>> import _skimage2 as ski2
            """
            return cls()

    proxy = adapt_obj_doctest(Impl, shim_module='skimage.tests.example')

    proxy_create = vars(proxy).get('create')
    assert proxy_create is not None, 'classmethod should be on proxy, not inherited'
    assert 'import skimage as ski' in proxy_create.__func__.__doc__

    # Original unmodified
    assert '_skimage2' in Impl.create.__func__.__doc__


def test_adapt_obj_doctest_adapts_staticmethod_docstrings():
    class Impl:
        @staticmethod
        def helper(x):
            """Helpful helper.

            >>> from _skimage2 import data
            """
            return x * 2

    proxy = adapt_obj_doctest(Impl, shim_module='skimage.tests.example')

    proxy_helper = vars(proxy).get('helper')
    assert proxy_helper is not None, 'staticmethod should be on proxy, not inherited'
    assert 'from skimage import data' in proxy_helper.__func__.__doc__

    # Original unmodified — staticmethod descriptor returns raw function
    assert '_skimage2' in Impl.helper.__doc__


def test_adapt_obj_doctest_class_method_without_skimage_ref_not_copied():
    """Methods without _skimage2 references stay inherited, not overridden."""

    class Impl:
        """Example class.

        >>> from _skimage2 import data
        """

        def plain(self):
            """No namespace references."""
            return 1

    proxy = adapt_obj_doctest(Impl, shim_module='skimage.tests.example')

    # 'plain' should NOT appear in the proxy's own dict — it is inherited
    assert 'plain' not in vars(proxy)
    assert proxy.plain.__doc__ == 'No namespace references.'


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

    proxy = adapt_obj_doctest(Impl, shim_module='skimage.tests.example')

    # __init__ docstring adapted
    proxy_init = vars(proxy).get('__init__')
    assert proxy_init is not None
    assert 'import skimage as ski' in proxy_init.__doc__

    # Original unmodified
    assert '_skimage2' in Impl.__init__.__doc__
