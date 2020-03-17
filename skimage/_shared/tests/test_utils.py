import sys
from skimage._shared.utils import check_nD, deprecate_kwarg
import numpy.testing as npt
import numpy as np
from skimage._shared import testing
import pytest


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


if __name__ == "__main__":
    npt.run_module_suite()
