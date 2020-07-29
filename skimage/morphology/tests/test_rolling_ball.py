"""
Tests for Rolling Ball Filter
(skimage.morphology.rolling_ball)

Author: Sebastian Wallkötter
"""

import numpy as np

from skimage import data
from skimage.morphology import rolling_ball

from skimage._shared.testing import assert_equal, fetch
from skimage._shared import testing


class TestRollingBall(object):
    def test_white_background(self):
        pass

    def test_black_background(self):
        pass

    def test_int8_img(self):
        pass

    def test_float_img(self):
        img = data.coins().astype(np.float32)
        try:
            rolling_ball(img)
            assert False
        except ValueError:
            pass
