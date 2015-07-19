""" Test _warnings module
"""

import warnings
from numpy.testing import assert_warns
from .. import _warnings
from ... import data, img_as_ubyte, img_as_float


def test_all_warnings():
    def foo():
        warnings.warn(RuntimeWarning("bar"))

    with warnings.catch_warnings():
        warnings.simplefilter('once')
        foo()

    foo()

    with _warnings.all_warnings():
        assert_warns(RuntimeWarning, foo)


def test_always_warn():
    def foo():
        with _warnings.always_warn():
            warnings.warn(RuntimeWarning("bar"))

    warnings.simplefilter('once')
    assert_warns(RuntimeWarning, foo)

    assert_warns(RuntimeWarning, foo)


def test_expected_warnings():
    with _warnings.expected_warnings(['precision loss']):
        img_as_ubyte(img_as_float(data.coins()))

    def foo():
        warnings.warn(RuntimeWarning("bar"))

    with _warnings.expected_warnings(['bar']):
        foo()
