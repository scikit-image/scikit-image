import numpy as np
import pytest

from skimage import data
from skimage.util.crop import bounding_box_crop

rng = np.random.default_rng()


def test_2d_crop_1():
    data = rng.random((50, 50))
    out_data = bounding_box_crop(data, [(0, 25)])
    np.testing.assert_array_equal(out_data, data[:25, :])


def test_2d_crop_2():
    data = rng.random((50, 50))
    out_data = bounding_box_crop(data, [(0, 25)], axes=[1])
    np.testing.assert_array_equal(out_data, data[:, :25])


def test_copy():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out_data_without_copy = bounding_box_crop(data, [(0, 3)], copy=False)
    out_data_copy = bounding_box_crop(data, [(0, 3)], copy=True)
    data[0, 0] = 100
    assert out_data_without_copy[0, 0] == 100
    assert out_data_copy[0, 0] == 1


def test_2d_crop_3():
    data = rng.random((50, 50))
    out_data = bounding_box_crop(data, [(0, 25), (0, 30)], axes=[1, 0])
    np.testing.assert_array_equal(out_data, data[:30, :25])


def test_nd_crop():
    data = rng.random((50, 50, 50))
    out_data = bounding_box_crop(data, [(0, 25)])
    np.testing.assert_array_equal(out_data, data[:25, :, :])


def test_axes_invalid():
    data = np.empty((2, 3))
    with pytest.raises(ValueError):
        bounding_box_crop(data, [(0, 3)], axes=[2])


def test_axes_limit_invalid():
    data = np.empty((50, 50))
    with pytest.raises(ValueError):
        bounding_box_crop(data, [(0, 51)], axes=[0])
    