import pytest
from pathlib import Path


@pytest.fixture
def test_root_dir():
    # Data files for tests reside in 'tests/skimage2'
    # (subdirectory intentionally omitted)
    return Path(__file__).absolute().parent.parent / 'skimage2'
