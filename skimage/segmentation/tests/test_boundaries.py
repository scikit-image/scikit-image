import numpy as np
from skimage.segmentation import find_boundaries, mark_boundaries

from skimage._shared.testing import assert_array_equal, assert_allclose


white = (1, 1, 1)


def test_find_boundaries():
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:7, 2:7] = 1

    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    result = find_boundaries(image)
    assert_array_equal(result, ref)


def test_find_boundaries_bool():
    image = np.zeros((5, 5), dtype=np.bool)
    image[2:5, 2:5] = True

    ref = np.array([[False, False, False, False, False],
                    [False, False,  True,  True,  True],
                    [False,  True,  True,  True,  True],
                    [False,  True,  True, False, False],
                    [False,  True,  True, False, False]], dtype=np.bool)
    result = find_boundaries(image)
    assert_array_equal(result, ref)


def test_mark_boundaries():
    image = np.zeros((10, 10))
    label_image = np.zeros((10, 10), dtype=np.uint8)
    label_image[2:7, 2:7] = 1

    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    marked = mark_boundaries(image, label_image, color=white, mode='thick')
    result = np.mean(marked, axis=-1)
    assert_array_equal(result, ref)

    ref = np.array([[0, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [2, 2, 1, 1, 1, 1, 1, 2, 2, 0],
                    [2, 1, 1, 1, 1, 1, 1, 1, 2, 0],
                    [2, 1, 1, 2, 2, 2, 1, 1, 2, 0],
                    [2, 1, 1, 2, 0, 2, 1, 1, 2, 0],
                    [2, 1, 1, 2, 2, 2, 1, 1, 2, 0],
                    [2, 1, 1, 1, 1, 1, 1, 1, 2, 0],
                    [2, 2, 1, 1, 1, 1, 1, 2, 2, 0],
                    [0, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    marked = mark_boundaries(image, label_image, color=white,
                             outline_color=(2, 2, 2), mode='thick')
    result = np.mean(marked, axis=-1)
    assert_array_equal(result, ref)


def test_mark_boundaries_bool():
    image = np.zeros((10, 10), dtype=np.bool)
    label_image = np.zeros((10, 10), dtype=np.uint8)
    label_image[2:7, 2:7] = 1

    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    marked = mark_boundaries(image, label_image, color=white, mode='thick')
    result = np.mean(marked, axis=-1)
    assert_array_equal(result, ref)


def test_mark_boundaries_subpixel():
    labels = np.array([[0, 0, 0, 0],
                       [0, 0, 5, 0],
                       [0, 1, 5, 0],
                       [0, 0, 5, 0],
                       [0, 0, 0, 0]], dtype=np.uint8)
    np.random.seed(0)
    image = np.round(np.random.rand(*labels.shape), 2)
    marked = mark_boundaries(image, labels, color=white, mode='subpixel')
    marked_proj = np.round(np.mean(marked, axis=-1), 2)

    ref_result = np.array(
        [[0.55, 0.62, 0.72, 0.69, 0.61, 0.54, 0.50],
         [0.47, 0.57, 0.71, 1.00, 1.00, 1.00, 0.67],
         [0.38, 0.50, 0.66, 1.00, 0.45, 1.00, 0.92],
         [0.68, 1.00, 1.00, 1.00, 0.63, 1.00, 0.80],
         [0.96, 1.00, 0.39, 1.00, 0.79, 1.00, 0.51],
         [0.82, 1.00, 1.00, 1.00, 0.33, 1.00, 0.19],
         [0.48, 0.69, 0.95, 1.00, 0.07, 1.00, 0.13],
         [0.11, 0.41, 0.86, 1.00, 1.00, 1.00, 0.63],
         [0.02, 0.25, 0.78, 0.88, 0.85, 0.91, 0.95]]
    )
    assert_allclose(marked_proj, ref_result, atol=0.01)
