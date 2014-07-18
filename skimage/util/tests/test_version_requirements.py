"""Tests for the version requirment functions.

"""
import numpy as np
from numpy.testing import assert_raises, assert_equal
import nose
from skimage.util import version_requirements as vr


def test_get_module_version():
    assert vr.get_module_version('numpy')
    assert vr.get_module_version('scipy')
    assert_raises(ImportError, lambda: vr.get_module_version('fakenumpy'))


def test_is_installed():
    assert vr.is_installed('python', '>=2.6')
    assert not vr.is_installed('numpy', '<1.0')


def test_require():

    @vr.require('python', '>2.6')
    @vr.require('numpy', '>1.5')
    def foo():
        return 1

    assert_equal(foo(), 1)

    @vr.require('scipy', '<0.1')
    def bar():
        return 0

    assert_raises(ImportError, lambda: bar())

    @vr.require('numpy', '<1.0')
    def test_this():
        assert False

    assert_raises(nose.SkipTest, lambda: test_this()())


def test_get_module():
    assert_equal(vr.get_module('numpy'), np)
    assert_equal(vr.get_module('nose'), nose)

