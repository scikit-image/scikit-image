"""Tests for the version requirement functions.

"""
import numpy as np
from numpy.testing import assert_raises, assert_equal
import nose
from skimage._shared import version_requirements as version_req


def test_get_module_version():
    assert version_req.get_module_version('numpy')
    assert version_req.get_module_version('scipy')
    assert_raises(ImportError,
                  lambda: version_req.get_module_version('fakenumpy'))


def test_is_installed():
    assert version_req.is_installed('python', '>=2.6')
    assert not version_req.is_installed('numpy', '<1.0')


def test_require():

    # A function that only runs on Python >2.6 and numpy > 1.5 (should pass)
    @version_req.require('python', '>2.6')
    @version_req.require('numpy', '>1.5')
    def foo():
        return 1

    assert_equal(foo(), 1)

    # function that requires scipy < 0.1 (should fail)
    @version_req.require('scipy', '<0.1')
    def bar():
        return 0

    assert_raises(ImportError, lambda: bar())


def test_get_module():
    assert_equal(version_req.get_module('numpy'), np)
    assert_equal(version_req.get_module('nose'), nose)
