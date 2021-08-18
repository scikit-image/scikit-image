import numpy as np

from skimage import data
from skimage.util.crop import bounding_box_crop


def test_2d_crop_1():
    data = np.random.random((50, 50))
    out_data = bounding_box_crop(data, [(0, 25)])
    assert out_data.shape == (25, 50)


def test_2d_crop_2():
    data = np.random.random((50, 50))
    out_data = bounding_box_crop(data, [(0, 25)], axis=[1])
    assert out_data.shape == (50, 25)


def test_copy():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out_data_without_copy = bounding_box_crop(data, [(0, 3)], copy=False)
    out_data_copy = bounding_box_crop(data, [(0, 3)], copy=True)
    data[0, 0] = 100
    assert out_data_without_copy[0, 0] == 100
    assert out_data_copy[0, 0] == 1


def test_2d_crop_3():
    data = np.random.random((50, 50))
    out_data = bounding_box_crop(data, [(0, 25), (0, 30)], axis=[1, 0])
    assert out_data.shape == (30, 25)


def test_nd_crop():
    data = np.random.random((50, 50, 50))
    out_data = bounding_box_crop(data, [(0, 25)])
    assert out_data.shape == (25, 50, 50)
