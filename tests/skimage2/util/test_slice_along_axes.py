import numpy as np
import pytest

from _skimage2.util import slice_along_axes


def test_2d_crop_0():
    rng = np.random.RandomState(1561468935)
    data = rng.random((50, 50))
    out = slice_along_axes(data, [(0, 25)])
    np.testing.assert_array_equal(out, data[:25, :])


def test_2d_crop_1():
    rng = np.random.RandomState(519127872)
    data = rng.random((50, 50))
    out = slice_along_axes(data, [(0, 25), (0, 10)])
    np.testing.assert_array_equal(out, data[:25, :10])


def test_2d_crop_2():
    rng = np.random.RandomState(1239741995)
    data = rng.random((50, 50))
    out = slice_along_axes(data, [(0, 25), (0, 30)], axes=[1, 0])
    np.testing.assert_array_equal(out, data[:30, :25])


def test_2d_negative():
    rng = np.random.RandomState(78880634)
    data = rng.random((50, 50))
    out = slice_along_axes(data, [(5, -5), (6, -6)])
    np.testing.assert_array_equal(out, data[5:-5, 6:-6])


def test_copy():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out_without_copy = slice_along_axes(data, [(0, 3)], axes=[1], copy=False)
    out_copy = slice_along_axes(data, [(0, 3)], axes=[0], copy=True)
    assert out_without_copy.base is data
    assert out_copy.base is not data


def test_nd_crop():
    rng = np.random.RandomState(2392534812)
    data = rng.random((50, 50, 50))
    out = slice_along_axes(data, [(0, 25)], axes=[2])
    np.testing.assert_array_equal(out, data[:, :, :25])


def test_axes_invalid():
    data = np.empty((2, 3))
    with pytest.raises(ValueError):
        slice_along_axes(data, [(0, 3)], axes=[2])


def test_axes_limit_invalid():
    data = np.empty((50, 50))
    with pytest.raises(ValueError):
        slice_along_axes(data, [(0, 51)], axes=[0])


def test_too_many_axes():
    data = np.empty((10, 10))
    with pytest.raises(ValueError):
        slice_along_axes(data, [(0, 1), (0, 1), (0, 1)])


def test_empty_slices_copy_true_returns_independent_array():
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    original = data.copy()
    out = slice_along_axes(data, [], copy=True)
    np.testing.assert_array_equal(out, original)
    assert out.shape == data.shape
    assert out.dtype == data.dtype
    assert not np.shares_memory(data, out)
    out[0, 0] = 99
    np.testing.assert_array_equal(data, original)


def test_empty_slices_copy_false_shares_memory():
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    out = slice_along_axes(data, [], copy=False)
    assert np.shares_memory(data, out)


def test_empty_slices_default_copy_shares_memory():
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    out = slice_along_axes(data, [])
    assert np.shares_memory(data, out)
