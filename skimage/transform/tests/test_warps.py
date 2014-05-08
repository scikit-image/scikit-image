from numpy.testing import (assert_array_almost_equal, run_module_suite,
                           assert_array_equal, assert_raises)
import numpy as np
from scipy.ndimage import map_coordinates

from skimage.transform import (warp, warp_coords, rotate, resize, rescale,
                               AffineTransform,
                               ProjectiveTransform,
                               SimilarityTransform,
                               downscale_local_mean)
from skimage import transform as tf, data, img_as_float
from skimage.color import rgb2gray


def test_warp_tform():
    x = np.zeros((5, 5), dtype=np.double)
    x[2, 2] = 1
    theta = - np.pi / 2
    tform = SimilarityTransform(scale=1, rotation=theta, translation=(0, 4))

    x90 = warp(x, tform, order=1)
    assert_array_almost_equal(x90, np.rot90(x))

    x90 = warp(x, tform.inverse, order=1)
    assert_array_almost_equal(x90, np.rot90(x))


def test_warp_callable():
    x = np.zeros((5, 5), dtype=np.double)
    x[2, 2] = 1
    refx = np.zeros((5, 5), dtype=np.double)
    refx[1, 1] = 1

    shift = lambda xy: xy + 1

    outx = warp(x, shift, order=1)
    assert_array_almost_equal(outx, refx)


def test_warp_matrix():
    x = np.zeros((5, 5), dtype=np.double)
    x[2, 2] = 1
    refx = np.zeros((5, 5), dtype=np.double)
    refx[1, 1] = 1

    matrix = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])

    # _warp_fast
    outx = warp(x, matrix, order=1)
    assert_array_almost_equal(outx, refx)
    # check for ndimage.map_coordinates
    outx = warp(x, matrix, order=5)


def test_homography():
    x = np.zeros((5, 5), dtype=np.double)
    x[1, 1] = 1
    theta = -np.pi / 2
    M = np.array([[np.cos(theta), - np.sin(theta), 0],
                  [np.sin(theta),   np.cos(theta), 4],
                  [0,               0,             1]])

    x90 = warp(x,
               inverse_map=ProjectiveTransform(M).inverse,
               order=1)
    assert_array_almost_equal(x90, np.rot90(x))


def test_fast_homography():
    img = rgb2gray(data.lena()).astype(np.uint8)
    img = img[:, :100]

    theta = np.deg2rad(30)
    scale = 0.5
    tx, ty = 50, 50

    H = np.eye(3)
    S = scale * np.sin(theta)
    C = scale * np.cos(theta)

    H[:2, :2] = [[C, -S], [S, C]]
    H[:2, 2] = [tx, ty]

    tform = ProjectiveTransform(H)
    coords = warp_coords(tform.inverse, (img.shape[0], img.shape[1]))

    for order in range(4):
        for mode in ('constant', 'reflect', 'wrap', 'nearest'):
            p0 = map_coordinates(img, coords, mode=mode, order=order)
            p1 = warp(img, tform, mode=mode, order=order)

            # import matplotlib.pyplot as plt
            # f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
            # ax0.imshow(img)
            # ax1.imshow(p0, cmap=plt.cm.gray)
            # ax2.imshow(p1, cmap=plt.cm.gray)
            # ax3.imshow(np.abs(p0 - p1), cmap=plt.cm.gray)
            # plt.show()

            d = np.mean(np.abs(p0 - p1))
            assert d < 0.001


def test_rotate():
    x = np.zeros((5, 5), dtype=np.double)
    x[1, 1] = 1
    x90 = rotate(x, 90)
    assert_array_almost_equal(x90, np.rot90(x))


def test_rotate_resize():
    x = np.zeros((10, 10), dtype=np.double)

    x45 = rotate(x, 45, resize=False)
    assert x45.shape == (10, 10)

    x45 = rotate(x, 45, resize=True)
    # new dimension should be d = sqrt(2 * (10/2)^2)
    assert x45.shape == (14, 14)


def test_rescale():
    # same scale factor
    x = np.zeros((5, 5), dtype=np.double)
    x[1, 1] = 1
    scaled = rescale(x, 2, order=0)
    ref = np.zeros((10, 10))
    ref[2:4, 2:4] = 1
    assert_array_almost_equal(scaled, ref)

    # different scale factors
    x = np.zeros((5, 5), dtype=np.double)
    x[1, 1] = 1
    scaled = rescale(x, (2, 1), order=0)
    ref = np.zeros((10, 5))
    ref[2:4, 1] = 1
    assert_array_almost_equal(scaled, ref)


def test_resize2d():
    x = np.zeros((5, 5), dtype=np.double)
    x[1, 1] = 1
    resized = resize(x, (10, 10), order=0)
    ref = np.zeros((10, 10))
    ref[2:4, 2:4] = 1
    assert_array_almost_equal(resized, ref)


def test_resize3d_keep():
    # keep 3rd dimension
    x = np.zeros((5, 5, 3), dtype=np.double)
    x[1, 1, :] = 1
    resized = resize(x, (10, 10), order=0)
    ref = np.zeros((10, 10, 3))
    ref[2:4, 2:4, :] = 1
    assert_array_almost_equal(resized, ref)
    resized = resize(x, (10, 10, 3), order=0)
    assert_array_almost_equal(resized, ref)


def test_resize3d_resize():
    # resize 3rd dimension
    x = np.zeros((5, 5, 3), dtype=np.double)
    x[1, 1, :] = 1
    resized = resize(x, (10, 10, 1), order=0)
    ref = np.zeros((10, 10, 1))
    ref[2:4, 2:4] = 1
    assert_array_almost_equal(resized, ref)


def test_resize3d_bilinear():
    # bilinear 3rd dimension
    x = np.zeros((5, 5, 2), dtype=np.double)
    x[1, 1, 0] = 0
    x[1, 1, 1] = 1
    resized = resize(x, (10, 10, 1), order=1)
    ref = np.zeros((10, 10, 1))
    ref[1:5, 1:5, :] = 0.03125
    ref[1:5, 2:4, :] = 0.09375
    ref[2:4, 1:5, :] = 0.09375
    ref[2:4, 2:4, :] = 0.28125
    assert_array_almost_equal(resized, ref)


def test_swirl():
    image = img_as_float(data.checkerboard())

    swirl_params = {'radius': 80, 'rotation': 0, 'order': 2, 'mode': 'reflect'}
    swirled = tf.swirl(image, strength=10, **swirl_params)
    unswirled = tf.swirl(swirled, strength=-10, **swirl_params)

    assert np.mean(np.abs(image - unswirled)) < 0.01


def test_const_cval_out_of_range():
    img = np.random.randn(100, 100)
    cval = - 10
    warped = warp(img, AffineTransform(translation=(10, 10)), cval=cval)
    assert np.sum(warped == cval) == (2 * 100 * 10 - 10 * 10)


def test_warp_identity():
    lena = img_as_float(rgb2gray(data.lena()))
    assert len(lena.shape) == 2
    assert np.allclose(lena, warp(lena, AffineTransform(rotation=0)))
    assert not np.allclose(lena, warp(lena, AffineTransform(rotation=0.1)))
    rgb_lena = np.transpose(np.asarray([lena, np.zeros_like(lena), lena]),
                            (1, 2, 0))
    warped_rgb_lena = warp(rgb_lena, AffineTransform(rotation=0.1))
    assert np.allclose(rgb_lena, warp(rgb_lena, AffineTransform(rotation=0)))
    assert not np.allclose(rgb_lena, warped_rgb_lena)
    # assert no cross-talk between bands
    assert np.all(0 == warped_rgb_lena[:, :, 1])


def test_warp_coords_example():
    image = data.lena().astype(np.float32)
    assert 3 == image.shape[2]
    tform = SimilarityTransform(translation=(0, -10))
    coords = warp_coords(tform, (30, 30, 3))
    map_coordinates(image[:, :, 0], coords[:2])


def test_downscale_local_mean():
    image1 = np.arange(4 * 6).reshape(4, 6)
    out1 = downscale_local_mean(image1, (2, 3))
    expected1 = np.array([[  4.,   7.],
                          [ 16.,  19.]])
    assert_array_equal(expected1, out1)

    image2 = np.arange(5 * 8).reshape(5, 8)
    out2 = downscale_local_mean(image2, (4, 5))
    expected2 = np.array([[ 14. ,  10.8],
                          [  8.5,   5.7]])
    assert_array_equal(expected2, out2)


def test_invalid():
    assert_raises(ValueError, warp, np.ones((4, )), SimilarityTransform())
    assert_raises(ValueError, warp, np.ones((4, 3, 3, 3)),
                  SimilarityTransform())


def test_inverse():
    tform = SimilarityTransform(scale=0.5, rotation=0.1)
    inverse_tform = SimilarityTransform(matrix=np.linalg.inv(tform.params))
    image = np.arange(10 * 10).reshape(10, 10).astype(np.double)
    assert_array_equal(warp(image, inverse_tform), warp(image, tform.inverse))


def test_slow_warp_nonint_oshape():
    image = np.random.random((5, 5))

    assert_raises(ValueError, warp, image, lambda xy: xy,
                  output_shape=(13.1, 19.5))

    warp(image, lambda xy: xy, output_shape=(13.0001, 19.9999))


if __name__ == "__main__":
    run_module_suite()
