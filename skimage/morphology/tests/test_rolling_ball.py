"""
Tests for Rolling Ball Filter
(skimage.morphology.rolling_ball)
"""

import numpy as np

from skimage import data
from skimage.morphology import rolling_ball


def test_black_background():
    img = 23 * np.ones((100, 100), dtype=np.uint8)
    result = rolling_ball(img, radius=1)
    assert np.allclose(result, np.zeros_like(img))
