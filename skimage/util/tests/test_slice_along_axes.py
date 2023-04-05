import numpy as np
import pytest

from skimage.util import slice_along_axes


rng = np.random.default_rng()


def test_2d_crop():
    data = rng.random((50, 50))
    out_data = slice_along_axes(data, [(0, 25)], axes=[1])
    np.testing.assert_array_equal(out_data, data[:, :25])


def test_2d_crop_2():
    data = rng.random((50, 50))
    out_data = slice_along_axes(data, [(0, 25), (0, 30)], axes=[1, 0])
    np.testing.assert_array_equal(out_data, data[:30, :25])


def test_copy():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out_data_without_copy = slice_along_axes(data, [(0, 3)], axes=[1], copy=False)
    out_data_copy = slice_along_axes(data, [(0, 3)], axes=[0], copy=True)
    assert out_data_without_copy.base is data
    assert out_data_copy.base is not data


def test_nd_crop():
    data = rng.random((50, 50, 50))
    out_data = slice_along_axes(data, [(0, 25)], axes=[2])
    np.testing.assert_array_equal(out_data, data[:, :, :25])


def test_axes_invalid():
    data = np.empty((2, 3))
    with pytest.raises(ValueError):
        slice_along_axes(data, [(0, 3)], axes=[2])


def test_axes_limit_invalid():
    data = np.empty((50, 50))
    with pytest.raises(ValueError):
        slice_along_axes(data, [(0, 51)], axes=[0])
