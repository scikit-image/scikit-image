import pytest
from skimage._shared.testing import setup_test, teardown_test

# List of files that pytest should ignore
collect_ignore = [
    "io/_plugins",
]


def pytest_runtest_setup(item):
    setup_test()


def pytest_runtest_teardown(item):
    teardown_test()

@pytest.fixture(autouse=True)
def handle_np2():
    # TODO: remove when we require numpy >= 2
    try:
        import numpy as np

        np.set_printoptions(legacy="1.21")
    except ImportError:
        pass
