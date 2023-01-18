from skimage._shared.testing import setup_test, teardown_test

# List of files that pytest should ignore
collect_ignore = [
    "io/_plugins",
    "future/graph",  # Remove after v0.20 release
]


def pytest_runtest_setup(item):
    setup_test()


def pytest_runtest_teardown(item):
    teardown_test()
