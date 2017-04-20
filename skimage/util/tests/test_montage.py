from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
import pytest

import numpy as np
from skimage.util.montage import montage2d, montage_rgb


def test_simple():
    n_images = 3
    height, width = 2, 3,
    arr_in = np.arange(n_images * height * width, dtype='float')
    arr_in = arr_in.reshape(n_images, height, width)

    arr_out = montage2d(arr_in)

    gt = np.array(
        [[  0. ,   1. ,   2. ,   6. ,   7. ,   8. ],
         [  3. ,   4. ,   5. ,   9. ,  10. ,  11. ],
         [ 12. ,  13. ,  14. ,   8.5,   8.5,   8.5],
         [ 15. ,  16. ,  17. ,   8.5,   8.5,   8.5]]
    )

    assert_array_equal(arr_out, gt)


def test_fill():
    n_images = 3
    height, width = 2, 3,
    arr_in = np.arange(n_images * height * width)
    arr_in = arr_in.reshape(n_images, height, width)

    arr_out = montage2d(arr_in, fill=0)

    gt = np.array(
        [[  0. ,   1. ,   2. ,   6. ,   7. ,   8. ],
         [  3. ,   4. ,   5. ,   9. ,  10. ,  11. ],
         [ 12. ,  13. ,  14. ,   0. ,   0. ,   0. ],
         [ 15. ,  16. ,  17. ,   0. ,   0. ,   0. ]]
    )

    assert_array_equal(arr_out, gt)


def test_shape():
    n_images = 15
    height, width = 11, 7
    arr_in = np.arange(n_images * height * width)
    arr_in = arr_in.reshape(n_images, height, width)

    alpha = int(np.ceil(np.sqrt(n_images)))

    arr_out = montage2d(arr_in)
    assert_equal(arr_out.shape, (alpha * height, alpha * width))


def test_grid_shape():
    n_images = 6
    height, width = 2, 2
    arr_in = np.arange(n_images * height * width, dtype=np.float32)
    arr_in = arr_in.reshape(n_images, height, width)
    arr_out = montage2d(arr_in, grid_shape=(3,2))
    correct_arr_out = np.array(
	[[  0.,   1.,   4.,   5.],
	 [  2.,   3.,   6.,   7.],
	 [  8.,   9.,  12.,  13.],
	 [ 10.,  11.,  14.,  15.],
	 [ 16.,  17.,  20.,  21.],
	 [ 18.,  19.,  22.,  23.]]
    )
    assert_array_equal(arr_out, correct_arr_out)


def test_rescale_intensity():
    n_images = 4
    height, width = 3, 3
    arr_in = np.arange(n_images * height * width, dtype=np.float32)
    arr_in = arr_in.reshape(n_images, height, width)

    arr_out = montage2d(arr_in, rescale_intensity=True)

    gt = np.array(
        [[ 0.   ,  0.125,  0.25 ,  0.   ,  0.125,  0.25 ],
         [ 0.375,  0.5  ,  0.625,  0.375,  0.5  ,  0.625],
         [ 0.75 ,  0.875,  1.   ,  0.75 ,  0.875,  1.   ],
         [ 0.   ,  0.125,  0.25 ,  0.   ,  0.125,  0.25 ],
         [ 0.375,  0.5  ,  0.625,  0.375,  0.5  ,  0.625],
         [ 0.75 ,  0.875,  1.   ,  0.75 ,  0.875,  1.   ]]
        )

    assert_equal(arr_out.min(), 0.0)
    assert_equal(arr_out.max(), 1.0)
    assert_array_equal(arr_out, gt)


def test_simple_padding():
    n_images = 2
    height, width = 2, 2,
    arr_in = np.arange(n_images * height * width)
    arr_in = arr_in.reshape(n_images, height, width)

    arr_out = montage2d(arr_in, padding_width=1)

    gt = np.array(
        [[0, 1, 0, 4, 5, 0],
         [2, 3, 0, 6, 7, 0],
         [0, 0, 0, 0, 0, 0],
         [3, 3, 3, 3, 3, 3],
         [3, 3, 3, 3, 3, 3],
         [3, 3, 3, 3, 3, 3]]
    )

    assert_array_equal(arr_out, gt)


def test_simple_rgb():

    n_images = 2
    height, width, n_channels = 2, 2, 2
    arr_in = np.arange(n_images * height * width * n_channels)
    arr_in = arr_in.reshape(n_images, height, width, n_channels)

    arr_out = montage_rgb(arr_in)

    gt = np.array(
        [[[ 0,  1],
          [ 2,  3],
          [ 8,  9],
          [10, 11]],
         [[ 4,  5],
          [ 6,  7],
          [12, 13],
          [14, 15]],
         [[ 7,  8],
          [ 7,  8],
          [ 7,  8],
          [ 7,  8]],
         [[ 7,  8],
          [ 7,  8],
          [ 7,  8],
          [ 7,  8]]]
        )
    assert_array_equal(arr_out, gt)

def test_error_ndim():
    arr_error = np.random.randn(1, 2, 3, 4)
    with pytest.raises(AssertionError):
        montage2d(arr_error)


def test_error_ndim_rgb_toosmall():
    arr_error = np.random.randn(1, 2, 3)
    with pytest.raises(AssertionError):
        montage_rgb(arr_error)


def test_error_ndim_rgb_toobig():
    arr_error = np.random.randn(1, 2, 3, 4, 5)
    with pytest.raises(AssertionError):
        montage_rgb(arr_error)


def test_error_ndim():
    arr_error = np.random.randn(1, 2, 3, 4)
    with pytest.raises(AssertionError):
        montage2d(arr_error)


if __name__ == '__main__':
    np.testing.run_module_suite()
