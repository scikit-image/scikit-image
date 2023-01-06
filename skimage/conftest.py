from skimage._shared.testing import setup_test, teardown_test

# List of files that pytest should ignore
collect_ignore = ["io/_plugins",]


def pytest_runtest_setup(item):
    setup_test()


def pytest_runtest_teardown(item):
    teardown_test()
