import pytest
import numpy as np
import imageio


@pytest.fixture(scope="session")
def random_gif_path(tmpdir_factory):
    """Create "random.gif" once per session and return its path."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, (24, 25, 14, 3), dtype=np.uint8)
    tmp_path = str(tmpdir_factory.mktemp("session-data").join("random.gif"))
    imageio.v3.imwrite(tmp_path, img)
    return tmp_path
