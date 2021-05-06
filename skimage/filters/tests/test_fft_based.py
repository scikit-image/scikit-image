import pytest
import numpy as np
from skimage.filters import butterworth


def test_butterworth():
    im = np.zeros((4, 4))
    filtered = butterworth(im)
    assert filtered.shape == im.shape
    assert np.all(im == filtered)


@pytest.mark.xfail(strict=True, raises=ValueError)
def test_butterworth_fail():
    im = np.zeros((4, 4, 2))
    butterworth(im)
