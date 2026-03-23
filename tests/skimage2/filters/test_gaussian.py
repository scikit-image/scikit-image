import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose

from skimage._shared.utils import _supported_float_type
import _skimage2 as ski2
from _skimage2.filters import gaussian


# skimage2.filters.gaussian ------------------------------------------------------------


def test_gaussian_negative_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1
    with pytest.raises(ValueError):
        gaussian(a, sigma=-1.0)
    with pytest.raises(ValueError):
        gaussian(a, sigma=[-1.0, 1.0])
    with pytest.raises(ValueError):
        gaussian(a, sigma=np.asarray([-1.0, 1.0]))


def test_gaussian_null_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.0
    assert np.all(gaussian(a, sigma=0) == a)


def test_gaussian_default_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.0
    assert_equal(
        gaussian(a), gaussian(a, sigma=1)
    )


@pytest.mark.parametrize(
    'dtype', [np.uint8, np.int32, np.float16, np.float32, np.float64]
)
def test_gaussian_image_dtype(dtype):
    a = np.zeros((3, 3), dtype=dtype)
    assert gaussian(a).dtype == _supported_float_type(a.dtype)


def test_gaussian_energy_decrease():
    a = np.zeros((3, 3))
    a[1, 1] = 1.0
    gaussian_a = gaussian(a, sigma=1, mode='reflect')
    assert gaussian_a.std() < a.std()


@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_gaussian_multichannel(channel_axis):
    a = np.zeros((5, 5, 3))
    a[1, 1] = np.arange(1, 4)
    a = np.moveaxis(a, -1, channel_axis)
    gaussian_rgb_a = gaussian(
        a, sigma=1, mode='reflect', channel_axis=channel_axis
    )
    # Check that the mean value is conserved in each channel
    # (color channels are not mixed together)
    spatial_axes = tuple([ax for ax in range(a.ndim) if ax != channel_axis % a.ndim])
    assert_allclose(
        a.mean(axis=spatial_axes), gaussian_rgb_a.mean(axis=spatial_axes)
    )

    if channel_axis % a.ndim == 2:
        # Check that the mean value is conserved in each channel
        # (color channels are not mixed together)
        assert_allclose(
            a.mean(axis=spatial_axes), gaussian_rgb_a.mean(axis=spatial_axes)
        )
    # Iterable sigma
    gaussian_rgb_a = gaussian(
        a, sigma=[1, 2], mode='reflect', channel_axis=channel_axis
    )
    assert_allclose(
        a.mean(axis=spatial_axes), gaussian_rgb_a.mean(axis=spatial_axes)
    )


def test_gaussian_migration_advice():
    """Test if legacy range scaling behavior is recoverable.

    Ported test that previously tested `preserve_range` parameter.
    """
    ones = np.ones((2, 2), dtype=np.int64)
    filtered_ones = gaussian(ski2.util.rescale_legacy(ones))
    assert_equal(filtered_ones, filtered_ones[0, 0])
    assert filtered_ones[0, 0] < 1e-10

    filtered_preserved = gaussian(ones)
    assert_equal(filtered_preserved, 1.0)

    img = np.array([[10.0, -10.0], [-4, 3]], dtype=np.float32)
    gaussian(img, sigma=1)


def test_gaussian_1d_ok():
    """Testing Gaussian Filter for 1D array.

    With any array consisting of positive integers and only one zero - it
    should filter all values to be greater than 0.1
    """
    nums = np.arange(7)
    filtered = gaussian(nums)
    assert np.all(filtered > 0.1)


def test_gaussian_4d_ok():
    img = np.zeros((5,) * 4)
    img[2, 2, 2, 2] = 1
    res = gaussian(img, sigma=1, mode='reflect')
    assert_allclose(res.sum(), 1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gaussian_preserve_output(dtype):
    image = np.arange(9, dtype=dtype).reshape((3, 3))
    out = np.zeros_like(image, dtype=dtype)
    gaussian_image = gaussian(image, sigma=1, out=out)
    assert gaussian_image is out


def test_gaussian_output_error():
    image = np.arange(9, dtype=np.float32).reshape((3, 3))
    out = np.zeros_like(image, dtype=np.uint8)
    with pytest.raises(ValueError, match="dtype of `out` must be float"):
        gaussian(image, sigma=1, out=out)
