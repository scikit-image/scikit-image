import pytest
from pathlib import Path


@pytest.fixture
def test_root_dir():
    return Path(__file__).absolute().parent

try:
    import pytest_run_parallel  # noqa:F401

    PARALLEL_RUN_AVAILABLE = True
except Exception:
    PARALLEL_RUN_AVAILABLE = False


def pytest_configure(config):
    if not PARALLEL_RUN_AVAILABLE:
        config.addinivalue_line(
            "markers",
            "thread_unsafe: mark the test function as single-threaded",
        )
