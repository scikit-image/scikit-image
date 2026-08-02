import pytest
from pathlib import Path

from _skimage2._shared.testing import local_data_path as _local_data_path


@pytest.fixture
def test_root_dir():
    # Data files for tests reside in 'tests/skimage2'
    # (subdirectory intentionally omitted)
    return Path(__file__).absolute().parent


@pytest.fixture
def local_data_path(test_root_dir):
    """Callable fixture: resolve a path (relative to `test_root_dir`) to a
    git-tracked test-data file, skipping the test if it's missing."""

    def get(relative_path):
        return _local_data_path(test_root_dir / relative_path)

    return get
