from skimage._docutils import adapt_doctest_doc, bind_public, bind_namespace


def test_adapt_import_lines():
    doc = """\
Examples
--------
>>> from _skimage2 import data
>>> from _skimage2.transform import resize
>>> import _skimage2 as ski2
"""
    adapted = adapt_doctest_doc(doc)
    assert 'from skimage import data' in adapted
    assert 'from skimage.transform import resize' in adapted
    assert 'import skimage as ski2' in adapted
    assert '_skimage2' not in adapted


def test_adapt_expected_output():
    doc = """\
>>> ski2.util.lookfor('regular_grid')
Search results for 'regular_grid'
---------------------------------
_skimage2.util.regular_grid
    Find `n_points` regularly spaced along `ar_shape`.
"""
    adapted = adapt_doctest_doc(doc)
    assert 'skimage.util.regular_grid' in adapted
    assert '_skimage2' not in adapted


def test_prose_outside_doctests_unchanged():
    doc = """\
See :func:`skimage2.util.img_as_float` for details.

Examples
--------
>>> from _skimage2 import data
"""
    adapted = adapt_doctest_doc(doc)
    assert ':func:`skimage2.util.img_as_float`' in adapted
    assert 'from skimage import data' in adapted


def test_keep_doctest_marker():
    doc = """\
>>> import skimage as ski1  #: skimage-shim-keep-doctest
>>> import skimage2 as ski2
>>> ski2.util.lookfor('x')
Search results for 'x'
---------------------------------
_skimage2.util.regular_grid
"""
    adapted = adapt_doctest_doc(doc)
    assert adapted == doc


def test_keep_doctest_marker_only_on_first_prompt():
    doc = """\
>>> from _skimage2 import first
>>> import _skimage2 as second  #: skimage-shim-keep-doctest
"""
    adapted = adapt_doctest_doc(doc)
    assert 'from skimage import first' in adapted
    assert 'import skimage as second' in adapted


def test_bind_public_sets_doc():
    def func():
        """Example

        >>> from _skimage2 import data
        """
        return 1

    bound = bind_public(func, shim_module='skimage.tests.example')
    assert 'from skimage import data' in bound.__doc__
    assert bound.__module__ == 'skimage.tests.example'
    assert bound() == 1


def test_bind_public_class_uses_proxy_without_mutating_impl():
    class Impl:
        """Example

        >>> from _skimage2 import data
        """

    bound = bind_public(Impl, shim_module='skimage.tests.example')
    assert bound is not Impl
    assert issubclass(bound, Impl)
    assert 'from skimage import data' in bound.__doc__
    assert bound.__module__ == 'skimage.tests.example'
    assert '_skimage2' in (Impl.__doc__ or '')
    assert Impl.__module__ == __name__


def test_bind_namespace():
    def one():
        """>>> from _skimage2 import data"""

    def two():
        """>>> import _skimage2 as ski"""

    ns = {'one': one, 'two': two, '__all__': ['one', 'two']}
    bind_namespace(ns)
    assert 'from skimage import data' in ns['one'].__doc__
    assert 'import skimage as ski' in ns['two'].__doc__


def test_adapt_preserves_array_output_block():
    doc = """\
>>> A = 1
>>> A
array([1])
>>> B = 2
"""
    adapted = adapt_doctest_doc(doc)
    assert 'array([1])' in adapted
