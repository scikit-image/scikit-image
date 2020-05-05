import sys
import pytest
import numpy as np
import numpy.testing as npt
from skimage._shared.utils import (check_nD, deprecate_kwarg,
                                   _validate_interpolation_order,
                                   change_default_value)
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings


def test_change_default_value():

    @change_default_value('arg1', new_value=-1, changed_version='0.12')
    def foo(arg0, arg1=0, arg2=1):
        """Expected docstring"""
        return arg0, arg1, arg2

    @change_default_value('arg1', new_value=-1, changed_version='0.12',
                          warning_msg="Custom warning message")
    def bar(arg0, arg1=0, arg2=1):
        """Expected docstring"""
        return arg0, arg1, arg2

    # Assert warning messages
    with pytest.warns(FutureWarning) as record:
        assert foo(0) == (0, 0, 1)
        assert bar(0) == (0, 0, 1)

    expected_msg = ("The new recommended value for arg1 is -1. Until "
                    "version 0.12, the default arg1 value is 0. From "
                    "version 0.12, the arg1 default value will be -1. "
                    "To avoid this warning, please explicitly set arg1 value.")

    assert str(record[0].message) == expected_msg
    assert str(record[1].message) == "Custom warning message"

    # Assert that nothing happens if arg1 is set
    with pytest.warns(None) as record:
        # No kwargs
        assert foo(0, 2) == (0, 2, 1)
        assert foo(0, arg1=0) == (0, 0, 1)

        # Function name and doc is preserved
        assert foo.__name__ == 'foo'
        if sys.flags.optimize < 2:
            # if PYTHONOPTIMIZE is set to 2, docstrings are stripped
            assert foo.__doc__ == 'Expected docstring'

    # Assert no warning was raised
    assert not record.list


def test_deprecated_kwarg():

    @deprecate_kwarg({'old_arg1': 'new_arg1'})
    def foo(arg0, new_arg1=1, arg2=None):
        """Expected docstring"""
        return arg0, new_arg1, arg2

    @deprecate_kwarg({'old_arg1': 'new_arg1'},
                     warning_msg="Custom warning message")
    def bar(arg0, new_arg1=1, arg2=None):
        """Expected docstring"""
        return arg0, new_arg1, arg2

    # Assert that the DeprecationWarning is raised when the deprecated
    # argument name is used and that the reasult is valid
    with pytest.warns(FutureWarning) as record:
        assert foo(0, old_arg1=1) == (0, 1, None)
        assert bar(0, old_arg1=1) == (0, 1, None)

    msg = ("'old_arg1' is a deprecated argument name "
           "for `foo`. Please use 'new_arg1' instead.")
    assert str(record[0].message) == msg
    assert str(record[1].message) == "Custom warning message"

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
        # Function name and doc is preserved
        assert foo.__name__ == 'foo'
        if sys.flags.optimize < 2:
            # if PYTHONOPTIMIZE is set to 2, docstrings are stripped
            assert foo.__doc__ == 'Expected docstring'

    # Assert no warning was raised
    assert not record.list


def test_check_nD():
    z = np.random.random(200**2).reshape((200, 200))
    x = z[10:30, 30:10]
    with testing.raises(ValueError):
        check_nD(x, 2)


@pytest.mark.parametrize('dtype', [bool, int, np.uint8, np.uint16,
                                   float, np.float32, np.float64])
@pytest.mark.parametrize('order', [None, -1, 0, 1, 2, 3, 4, 5, 6])
def test_validate_interpolation_order(dtype, order):
    if order is None:
        # Default order
        assert (_validate_interpolation_order(dtype, None) == 0
                if dtype == bool else 1)
    elif order < 0 or order > 5:
        # Order not in valid range
        with testing.raises(ValueError):
            _validate_interpolation_order(dtype, order)
    elif dtype == bool and order != 0:
        # Deprecated order for bool array
        with expected_warnings(["Input image dtype is bool"]):
            assert _validate_interpolation_order(bool, order) == order
    else:
        # Valid use case
        assert _validate_interpolation_order(dtype, order) == order


if __name__ == "__main__":
    npt.run_module_suite()
