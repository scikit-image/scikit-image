import numpy as np
import pytest
from numpy.testing import assert_array_equal

from skimage import rescale_as_float
from skimage.util.compare import compare_images


def test_compate_images_ValueError_shape():
    img1 = np.zeros((10, 10), dtype=np.uint8)
    img2 = np.zeros((10, 1), dtype=np.uint8)
    with pytest.raises(ValueError):
        compare_images(img1, img2)


@pytest.mark.parametrize('rescale_input', [False, True])
def test_compare_images_diff(rescale_input):
    img1 = np.zeros((10, 10), dtype=np.uint8)
    img1[3:8, 3:8] = 255
    img2 = np.zeros_like(img1)
    img2[3:8, 0:8] = 255
    if rescale_input:
        img1 = rescale_as_float(img1)
        img2 = rescale_as_float(img2)
    expected_result = np.zeros_like(img1, dtype=np.float64)
    expected_result[3:8, 0:3] = img1.max()
    result = compare_images(img1, img2, method='diff')
    assert_array_equal(result, expected_result)


@pytest.mark.parametrize('rescale_input', [False, True])
def test_compare_images_blend(rescale_input):
    img1 = np.zeros((10, 10), dtype=np.uint8)
    img1[3:8, 3:8] = 255
    img2 = np.zeros_like(img1)
    img2[3:8, 0:8] = 255
    if rescale_input:
        img1 = rescale_as_float(img1)
        img2 = rescale_as_float(img2)
    expected_result = np.zeros_like(img1, dtype=np.float64)
    imax = img1.max()
    expected_result[3:8, 3:8] = imax
    expected_result[3:8, 0:3] = imax / 2
    result = compare_images(img1, img2, method='blend')
    assert_array_equal(result, expected_result)


@pytest.mark.parametrize('rescale_input', [False, True])
def test_compare_images_checkerboard_default(rescale_input):
    img1 = np.zeros((2**4, 2**4), dtype=np.uint8)
    img2 = np.full(img1.shape, fill_value=255, dtype=np.uint8)
    if rescale_input:
        img1 = rescale_as_float(img1)
        img2 = rescale_as_float(img2)
    res = compare_images(img1, img2, method='checkerboard')
    imax = img2.max()
    exp_row1 = np.array(
        [0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1.]
    ) * imax
    exp_row2 = np.array(
        [1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.]
    ) * imax
    for i in (0, 1, 4, 5, 8, 9, 12, 13):
        assert_array_equal(res[i, :], exp_row1)
    for i in (2, 3, 6, 7, 10, 11, 14, 15):
        assert_array_equal(res[i, :], exp_row2)


@pytest.mark.parametrize('rescale_input', [False, True])
def test_compare_images_checkerboard_tuple(rescale_input):
    img1 = np.zeros((2**4, 2**4), dtype=np.uint8)
    img2 = np.full(img1.shape, fill_value=255, dtype=np.uint8)
    if rescale_input:
        img1 = rescale_as_float(img1)
        img2 = rescale_as_float(img2)
    res = compare_images(img1, img2, method='checkerboard', n_tiles=(4, 8))
    imax = img2.max()
    exp_row1 = np.array(
        [0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1.]
    ) * imax
    exp_row2 = np.array(
        [1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.]
    ) * imax
    for i in (0, 1, 2, 3, 8, 9, 10, 11):
        assert_array_equal(res[i, :], exp_row1)
    for i in (4, 5, 6, 7, 12, 13, 14, 15):
        assert_array_equal(res[i, :], exp_row2)
