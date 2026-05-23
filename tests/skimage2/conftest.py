import pytest
from pathlib import Path


@pytest.fixture
def test_root_dir():
    return Path(__file__).absolute().parent
