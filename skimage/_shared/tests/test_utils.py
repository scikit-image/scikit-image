from skimage._shared.utils import (copy_func, assert_nD, interpret_arg)
import numpy.testing as npt
import numpy as np
import pytest


def test_assert_nD():
    z = np.random.random(200**2).reshape((200, 200))
    x = z[10:30, 30:10]
    with pytest.raises(ValueError):
        assert_nD(x, 2)


def test_copyfunc():
    def foo(a):
        return a

    bar = copy_func(foo, name='bar')
    other = copy_func(foo)

    npt.assert_equal(bar.__name__, 'bar')
    npt.assert_equal(other.__name__, 'foo')

    other.__name__ = 'other'

    npt.assert_equal(foo.__name__, 'foo')


def test_interpret_arg():
    standardized_output = interpret_arg(None, 3)
    npt.assert_array_equal(standardized_output, np.zeros(3))

    standardized_output = interpret_arg((None, 10), 5, default=1)
    npt.assert_array_equal(standardized_output, np.array([1, 10, 1, 1, 1]))


if __name__ == "__main__":
    npt.run_module_suite()
