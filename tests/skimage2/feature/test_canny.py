import numpy as np
import pytest

from _skimage2.feature import canny


@pytest.mark.parametrize(
    'mode, should_match',
    [
        ('nearest', True),
        ('reflect', False),
        ('constant', False),
        ('mirror', False),
        ('wrap', False),
    ],
)
def test_default_mode_is_nearest(mode, should_match):
    rng = np.random.default_rng(0)
    image = rng.random((64, 64))
    result_default = canny(image)
    result_other = canny(image, mode=mode)
    assert np.array_equal(result_default, result_other) == should_match
