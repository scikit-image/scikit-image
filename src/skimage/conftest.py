"""
This conftest is required to set the numpy print options
to legacy mode for doctests, and to define a `thread_unsafe` pytest marker.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "thread_unsafe: mark the test function as single-threaded",
    )


@pytest.fixture(autouse=True)
def handle_np2():
    # TODO: remove when we require numpy >= 2
    try:
        import numpy as np

        np.set_printoptions(legacy="1.21")
    except ImportError:
        pass
