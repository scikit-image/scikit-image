import pytest
import numpy as np
import imageio.v3 as iio


@pytest.fixture(scope="session")
def random_gif_path(tmp_path_factory):
    """Create "random.gif" once per session and return its path."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, (24, 25, 14, 3), dtype=np.uint8)
    tmp_path = tmp_path_factory.mktemp("session-data") / ("random.gif")
    iio.imwrite(tmp_path, img)
    return str(tmp_path)
