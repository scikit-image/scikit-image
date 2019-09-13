from skimage._shared.utils import copy_func, check_nD, deprecate_arg
import numpy.testing as npt
import numpy as np
from skimage._shared import testing
import pytest


def test_deprecated_arg():

    @deprecate_arg({'old_arg1': 'new_arg1'})
    def foo(arg0, new_arg1=1, arg2=None):
        return arg0, new_arg1, arg2

    # Assert that the DeprecationWarning is raised when the deprecated
    # argument name is used and that the reasult is valid
    with pytest.warns(FutureWarning):
        assert foo(0, old_arg1=1) == (0, 1, None)

    # Assert that nothing happens when the function is called with the
    # new API
    with pytest.warns(None) as record:
        # No kwargs
        assert foo(0) == (0, 1, None)
        assert foo(0, 2) == (0, 2, None)
        assert foo(0, 1, 2) == (0, 1, 2)
        # Kwargs without deprecated argument
        assert foo(0, new_arg1=1, arg2=2) == (0, 1, 2)
        assert foo(0, new_arg1=2) == (0, 2, None)
        assert foo(0, arg2=2) == (0, 1, 2)
        assert foo(0, 1, arg2=2) == (0, 1, 2)

    # Assert no warning was raised
    assert not record.list


def test_check_nD():
    z = np.random.random(200**2).reshape((200, 200))
    x = z[10:30, 30:10]
    with testing.raises(ValueError):
        check_nD(x, 2)


def test_copyfunc():
    def foo(a):
        return a

    bar = copy_func(foo, name='bar')
    other = copy_func(foo)

    npt.assert_equal(bar.__name__, 'bar')
    npt.assert_equal(other.__name__, 'foo')

    other.__name__ = 'other'

    npt.assert_equal(foo.__name__, 'foo')


if __name__ == "__main__":
    npt.run_module_suite()
