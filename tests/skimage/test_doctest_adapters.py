import inspect
import sys
import types
import typing

import numpy as np

from skimage._doctest_adapters import (
    _adapt_doctest_doc,
    _adapt_obj_doctest,
    adapt_doctests,
    as_shim,
    returns_shim,
    shim_wraps,
)

import pytest


def skip_if_pyopt2(func):
    return pytest.mark.skipif(
        sys.flags.optimize >= 2,
        reason='PYTHONOPTIMIZE=2 strips docstrings',
    )(func)


def _as_impl(cls):
    """Give test class `cls` the module of a ``_skimage2`` implementation class."""
    cls.__module__ = '_skimage2.tests.example'
    return cls


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
    @_as_impl
    class Impl:
        """Example

        >>> from _skimage2 import data
        """

    assert Impl.__module__ == '_skimage2.tests.example'
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
@pytest.mark.thread_unsafe(reason="mutates sys.modules under a fixed module name")
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
    @_as_impl
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
    assert Impl.__module__ == '_skimage2.tests.example'


@skip_if_pyopt2
def test_adapt_obj_doctest_adapts_classmethod_docstrings():
    @_as_impl
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
    @_as_impl
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

    @_as_impl
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
    @_as_impl
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


def test_adapt_obj_doctest_always_wraps_class_without_doc_rewrite():
    """Classes are always wrapped so shim patches cannot mutate implementation"""

    @_as_impl
    class Impl:
        """Plain class with no skimage2 doctest imports."""

        def method(self):
            """Also plain."""
            return 1

    proxy = _adapt_obj_doctest(Impl, shim_module='skimage.tests.example')
    assert proxy is not Impl
    assert issubclass(proxy, Impl)
    assert proxy.__module__ == 'skimage.tests.example'
    assert proxy.__doc__ == Impl.__doc__
    # "method" defined in Impl, but not proxy class.
    assert 'method' in vars(Impl)
    assert 'method' not in vars(proxy)


def test_adapt_obj_doctest_always_wraps_class_with_none_doc():
    @_as_impl
    class Impl:
        pass

    Impl.__doc__ = None  # Simulates optimize doctest strips.
    proxy = _adapt_obj_doctest(Impl, shim_module='skimage.tests.example')
    assert proxy is not Impl
    assert issubclass(proxy, Impl)
    assert proxy.__module__ == 'skimage.tests.example'
    assert proxy.__doc__ is None


def test_adapt_obj_doctest_leaves_shim_local_class_unchanged():
    # Check class already owned by shim module does not get reshimmed.
    class AlreadyShimmed:
        """>>> from _skimage2 import data"""

    AlreadyShimmed.__module__ = 'skimage.tests.example'
    out = _adapt_obj_doctest(AlreadyShimmed, shim_module='skimage.tests.example')
    assert out is AlreadyShimmed


def test_shim_wraps():
    """shim_wraps copies wrapper metadata but keeps the enclosing __module__."""

    def impl(a, b=1):
        """Implementation docstring."""
        return a + b

    impl.__module__ = '_skimage2.tests.example'

    @shim_wraps(impl)
    def wrapper(*args, **kwargs):
        return impl(*args, **kwargs)

    assert wrapper.__module__ == __name__
    assert wrapper.__name__ == impl.__name__
    assert wrapper.__qualname__ == impl.__qualname__
    assert wrapper.__doc__ == impl.__doc__
    assert wrapper.__annotations__ == impl.__annotations__
    assert wrapper.__wrapped__ is impl
    assert wrapper(1, b=2) == 3
    assert str(inspect.signature(wrapper)) == str(inspect.signature(impl))


def test_shim_wraps_keeps_module_for_closures():
    """A wrapper defined inside a function still keeps the module __name__."""

    def impl():
        """Implementation docstring."""
        return 1

    impl.__module__ = '_skimage2.tests.example'

    def make():
        @shim_wraps(impl)
        def wrapper():
            return impl()

        return wrapper

    assert make().__module__ == __name__


def test_as_shim():
    @_as_impl
    class Impl:
        pass

    ns = {'Impl': Impl, '__name__': 'skimage.tests.example_as_shim'}
    adapt_doctests(ns)
    obj = Impl()
    assert as_shim(obj) is obj
    assert type(obj) is ns['Impl']
    items = as_shim([Impl(), Impl()])
    assert [type(item) for item in items] == [ns['Impl']] * 2
    other = object()
    assert as_shim(other) is other


def test_as_shim_refuses_shim_init():
    @_as_impl
    class Impl:
        pass

    ns = {'Impl': Impl, '__name__': 'skimage.tests.example_shim_init'}
    adapt_doctests(ns)

    def __init__(self):
        self.extra = 1

    ns['Impl'].__init__ = __init__
    with pytest.raises(TypeError, match='must not define __init__'):
        as_shim(Impl())


def test_as_shim_allows_doctest_copy_of_init():
    @_as_impl
    class Impl:
        def __init__(self):
            """Example.

            >>> import _skimage2 as ski2
            """
            self.value = 1

    ns = {'Impl': Impl, '__name__': 'skimage.tests.example_init_copy'}
    adapt_doctests(ns)
    assert '__init__' in vars(ns['Impl'])  # Copy with adapted docstring.
    assert type(as_shim(Impl())) is ns['Impl']


def test_returns_shim():
    @_as_impl
    class Impl:
        pass

    def make():
        """Implementation docstring."""
        return Impl()

    make.__module__ = '_skimage2.tests.example'
    ns = {'Impl': Impl, '__name__': 'skimage.tests.example_returns_shim'}
    adapt_doctests(ns)
    wrapper = returns_shim(make)
    assert type(wrapper()) is ns['Impl']
    assert wrapper.__module__ == __name__
    assert wrapper.__doc__ == make.__doc__
    assert wrapper.__wrapped__ is make


def test_adapt_doctests_wraps_class_without_mutating_impl_add():
    """Shim-only __add__ patch must not land on the implementation class."""

    class Impl:
        def __add__(self, other):
            return 'impl'

    Impl.__module__ = '_skimage2.tests.example'
    impl_add = Impl.__add__

    ns = {
        'Impl': Impl,
        '__name__': 'skimage.tests.example_shim',
    }
    adapt_doctests(ns)
    shim = ns['Impl']
    assert shim is not Impl
    shim.__add__ = lambda self, other: 'shim'
    assert Impl.__add__ is impl_add  # __add__ not changed in implementation.
    assert Impl().__add__(None) == 'impl'
    assert shim().__add__(None) == 'shim'


def test_geometric_shim_does_not_patch_skimage2_affine_add():
    """Importing skimage geometric shims must leave _skimage2.__add__ intact."""
    import _skimage2.transform._geometric as ski2_g
    import skimage.transform._geometric as ski_g

    Ski2Affine, Ski2Projective, Ski2Similarity = (
        ski2_g.AffineTransform,
        ski2_g.ProjectiveTransform,
        ski_g.SimilarityTransform,
    )

    assert ski_g.AffineTransform is not Ski2Affine
    assert ski_g.AffineTransform.__add__ is not Ski2Affine.__add__

    # Check can be combined (not so when shims not correctly adapted).
    tform1 = Ski2Affine(scale=(0.1, 0.1), rotation=0.3)
    tform2 = Ski2Similarity(scale=0.1, rotation=0.9)
    combined = tform1 + tform2
    assert isinstance(combined, Ski2Projective)


def test_adapt_doctests_matches_class_hierarchy():
    class Base:
        pass

    class Child(Base):
        pass

    class GrandChild(Child):
        pass

    for cls in (Base, Child, GrandChild):
        cls.__module__ = '_skimage2.tests.example'
    # Parents in another shim module, and listed after their children.
    ns_base = {'Base': Base, '__name__': 'skimage.tests.example_base'}
    ns = {
        'GrandChild': GrandChild,
        'Child': Child,
        '__name__': 'skimage.tests.example_shim',
    }
    adapt_doctests(ns_base)
    adapt_doctests(ns)
    ShimBase, ShimChild, ShimGrandChild = ns_base['Base'], ns['Child'], ns['GrandChild']
    assert issubclass(ShimChild, ShimBase)
    assert issubclass(ShimGrandChild, ShimChild)
    assert issubclass(ShimGrandChild, GrandChild)
    assert not issubclass(ShimBase, ShimChild)


def test_adapt_doctests_refuses_missing_parent_shim():
    class Base:
        pass

    class Child(Base):
        pass

    for cls in (Base, Child):
        cls.__module__ = '_skimage2.tests.example'
    # The child's module first: `Base` has no shim class yet.
    ns_child = {'Child': Child, '__name__': 'skimage.tests.example_child'}
    with pytest.raises(TypeError, match='No shim class for'):
        adapt_doctests(ns_child)


def test_adapt_doctests_does_not_shim_other_bases():
    class Other:
        pass

    class Impl(Other):
        pass

    Impl.__module__ = '_skimage2.tests.example'
    ns = {'Impl': Impl, '__name__': 'skimage.tests.example_other_base'}
    adapt_doctests(ns)
    assert ns['Impl'].__bases__ == (Impl,)


def test_adapt_doctests_shares_shim_class_between_modules():
    @_as_impl
    class Impl:
        pass

    ns1 = {'Impl': Impl, '__name__': 'skimage.tests.example_one'}
    ns2 = {'Impl': Impl, '__name__': 'skimage.tests.example_two'}
    adapt_doctests(ns1)
    adapt_doctests(ns2)
    assert ns1['Impl'] is not Impl
    assert ns2['Impl'] is ns1['Impl']


def test_adapt_doctests_leaves_skimage_classes_unchanged():
    class Native:
        pass

    Native.__module__ = 'skimage.tests.example_other'
    ns = {'Native': Native, '__name__': 'skimage.tests.example_shim'}
    adapt_doctests(ns)
    assert ns['Native'] is Native


def test_adapt_doctests_keeps_runtime_protocol():
    @_as_impl
    @typing.runtime_checkable
    class Proto(typing.Protocol):
        def run(self): ...

    class Impl:
        def run(self):
            pass

    ns = {'Proto': Proto, '__name__': 'skimage.tests.example_shim'}
    adapt_doctests(ns)
    assert ns['Proto'] is not Proto
    assert isinstance(Impl(), ns['Proto'])
    assert issubclass(Impl, ns['Proto'])
    assert not isinstance(object(), ns['Proto'])
