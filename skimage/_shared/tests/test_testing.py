"""Testing decorators module"""

import inspect
import re
import warnings

import pytest
from skimage._shared.testing import (
    run_in_parallel,
    assert_stacklevel,
)
from skimage._shared._dependency_checks import is_wasm

from skimage._shared._warnings import expected_warnings
from warnings import warn


@pytest.mark.skipif(is_wasm, reason="Cannot start threads in WASM")
def test_run_in_parallel():
    state = []

    @run_in_parallel()
    def change_state1():
        state.append(None)

    change_state1()
    assert len(state) == 2

    @run_in_parallel(num_threads=1)
    def change_state2():
        state.append(None)

    change_state2()
    assert len(state) == 3

    @run_in_parallel(num_threads=3)
    def change_state3():
        state.append(None)

    change_state3()
    assert len(state) == 6


@pytest.mark.skipif(is_wasm, reason="Cannot run parallel code in WASM")
def test_parallel_warning():
    @run_in_parallel()
    def change_state_warns_fails():
        warn("Test warning for test parallel", stacklevel=2)

    with expected_warnings(['Test warning for test parallel']):
        change_state_warns_fails()

    @run_in_parallel(warnings_matching=['Test warning for test parallel'])
    def change_state_warns_passes():
        warn("Test warning for test parallel", stacklevel=2)

    change_state_warns_passes()


def test_expected_warnings_noop():
    # This will ensure the line beolow it behaves like a no-op
    with expected_warnings(['Expected warnings test']):
        # This should behave as a no-op
        with expected_warnings(None):
            warn('Expected warnings test')


class Test_assert_stacklevel:
    def raise_warning(self, *args, **kwargs):
        warnings.warn(*args, **kwargs)

    def test_correct_stacklevel(self):
        # Should pass if stacklevel is set correctly
        with pytest.warns(UserWarning, match="passes") as record:
            self.raise_warning("passes", UserWarning, stacklevel=2)
        assert_stacklevel(record)

    @pytest.mark.parametrize("level", [1, 3])
    def test_wrong_stacklevel(self, level):
        # AssertionError should be raised for wrong stacklevel
        with pytest.warns(UserWarning, match="wrong") as record:
            self.raise_warning("wrong", UserWarning, stacklevel=level)
        # Check that message contains expected line on right side
        line_number = inspect.currentframe().f_lineno - 2
        regex = ".*" + re.escape(f"Expected: {__file__}:{line_number}")
        with pytest.raises(AssertionError, match=regex):
            assert_stacklevel(record, offset=-5)
