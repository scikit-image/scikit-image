import pytest
import numpy as np
from skimage.data import astronaut, coins
from skimage.filters import butterworth
from skimage._shared.testing import fetch


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
    [(0, np.float64), (1, np.complex128), (2, np.uint8), (3, np.int64)],
)
def test_butterworth_4D_channel(chan, dtype):
    im = np.zeros((3, 4, 5, 6), dtype=dtype)
    filtered = butterworth(im, channel_axis=chan)
    assert filtered.shape == im.shape


def test_butterworth_correctness_bw():
    small = coins()[180:210, 260:291]
    filtered = butterworth(small,
                           cutoff_frequency_ratio=0.2,
                           preserve_range=True).astype(np.uint8)
    path = fetch('filters/tests/coins_butter.npy')
    np.testing.assert_allclose(filtered, np.load(path))


def test_butterworth_correctness_rgb():
    small = astronaut()[135:155, 205:225]
    filtered = butterworth(small,
                           cutoff_frequency_ratio=0.3,
                           high_pass=True,
                           preserve_range=True,
                           channel_axis=-1).astype(np.uint8)
    path = fetch('filters/tests/astronaut_butter.npy')
    np.testing.assert_allclose(filtered, np.load(path))
