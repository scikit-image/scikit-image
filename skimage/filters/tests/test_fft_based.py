import pytest
import numpy as np
from skimage.filters import butterworth


def test_butterworth_2D():
    im = np.zeros((4, 4))
    filtered = butterworth(im)
    assert filtered.shape == im.shape
    assert np.all(im == filtered)


def test_butterworth_3D():
    im = np.zeros((3, 4, 5))
    filtered = butterworth(im)
    assert filtered.shape == im.shape
    assert np.all(im == filtered)


def test_butterworth_4D():
    im = np.zeros((3, 4, 5, 6))
    filtered = butterworth(im, preserve_range=True)
    assert filtered.shape == im.shape
    assert np.all(im == filtered)


@pytest.mark.parametrize(
        "chan",
        [0, 1, 2, 3]
        )
def test_butterworth_4D_channel(chan):
    im = np.zeros((3, 4, 5, 6))
    filtered = butterworth(im, channel_axis=chan)
    assert filtered.shape == im.shape
