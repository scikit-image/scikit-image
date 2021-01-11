import numpy as np

from skimage._shared import testing
from skimage._shared.testing import assert_equal

from skimage.util.points_to_labels import label_points


def test_label_points_coords_type():
    coords, image = [[1, 2], [3, 4]], np.zeros((5, 5))
    with testing.raises(TypeError):
        label_points(coords, image)


def test_label_points_coords_2D():
    coords, image = np.zeros((2, 3)), np.zeros((5, 5))
    with testing.raises(ValueError):
        label_points(coords, image)


def test_label_points_coords_shape():
    coords, image = np.zeros((2, 2, 1)), np.zeros((5, 5))
    with testing.raises(ValueError):
        label_points(coords, image)


def test_label_points_two_channel_image():
    coords, image = np.array([[0, 0],
                              [1, 1],
                              [2, 2],
                              [3, 3],
                              [4, 4]]), np.zeros((5, 5))
    mask = label_points(coords, image)
    assert_equal(mask, np.array([[1, 0, 0, 0, 0],
                                 [0, 2, 0, 0, 0],
                                 [0, 0, 3, 0, 0],
                                 [0, 0, 0, 4, 0],
                                 [0, 0, 0, 0, 5]]))


def test_label_points_three_channel_image():
    coords, image = np.array([[0, 0],
                              [1, 1],
                              [2, 2],
                              [3, 3],
                              [4, 4]]), np.zeros((5, 5, 3))
    mask = label_points(coords, image)
    assert_equal(mask, np.array([[1, 0, 0, 0, 0],
                                 [0, 2, 0, 0, 0],
                                 [0, 0, 3, 0, 0],
                                 [0, 0, 0, 4, 0],
                                 [0, 0, 0, 0, 5]]))
