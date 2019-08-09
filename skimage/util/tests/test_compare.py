import numpy as np
from skimage._shared.testing import assert_array_equal

from skimage.util.compare import compare_images


def test_compare_images_diff():
    img1 = np.zeros((10, 10), dtype=np.uint)
    img1[3:8, 3:8] = 1
    img2 = np.zeros_like(img1)
    img2[3:8, 0:8] = 1
    expected_result = np.zeros_like(img1)
    expected_result[3:8, 0:3] = 1
    result = compare_images(img1, img2, method='diff')
    assert_array_equal(result, expected_result)
