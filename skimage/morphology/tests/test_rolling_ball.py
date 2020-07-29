"""
Tests for Rolling Ball Filter
(skimage.morphology.rolling_ball)

Author: Sebastian Wallkötter
"""

import numpy as np

from skimage import data
from skimage.morphology import rolling_ball
from skimage.util import invert

from skimage._shared.testing import assert_equal, fetch
from skimage._shared import testing


class TestRollingBall(object):
    def test_white_background(self):
        img = 23 * np.ones((100, 100), dtype=np.uint8)
        img = invert(img)
        result, bg = rolling_ball(img, radius=1, white_background=True)
        assert np.allclose(result, 255 *np.ones_like(img))
        assert np.allclose(bg, img)

    def test_black_background(self):
        img = 23 * np.ones((100, 100), dtype=np.uint8)
        result, bg = rolling_ball(img, radius=1)
        assert np.allclose(result, np.zeros_like(img))
        assert np.allclose(bg, img)

    def test_float_img(self):
        img = data.coins().astype(np.float32)
        try:
            rolling_ball(img)
            assert False
        except ValueError:
            # currently not supported
            pass
