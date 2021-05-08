import pytest
import numpy as np
from .. import butterworth


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
    "chan, dtype",
    [(0, np.float), (1, np.complex128), (2, np.uint8), (3, np.complex256)],
)
def test_butterworth_4D_channel(chan, dtype):
    im = np.zeros((3, 4, 5, 6), dtype=dtype)
    filtered = butterworth(im, channel_axis=chan)
    assert filtered.shape == im.shape
