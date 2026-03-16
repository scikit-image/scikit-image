import numpy as np
import pytest

from skimage2 import feature


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
    result_default = feature.canny(image)
    result_other = feature.canny(image, mode=mode)
    assert np.array_equal(result_default, result_other) == should_match
