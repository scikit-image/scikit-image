import numpy as np
from scipy.ndimage import map_coordinates

from .._warp import (warp, _stackcopy, _linear_polar_mapping,
                     _log_polar_mapping, warp_coords)
from .._geometric import (AffineTransform, ProjectiveTransform,
                          SimilarityTransform)
from ..dtype import img_as_float
from skimage import data
from skimage.color import rgb2gray
from skimage.draw import circle_perimeter_aa
from skimage.feature import peak_local_max
from skimage._shared import testing
from skimage._shared.testing import (assert_almost_equal, assert_equal,
                                     test_parallel)
from skimage._shared._warnings import expected_warnings


np.random.seed(0)


def test_stackcopy():
    layers = 4
    x = np.empty((3, 3, layers))
    y = np.eye(3, 3)
    _stackcopy(x, y)
    for i in range(layers):
        assert_almost_equal(x[..., i], y)


def test_warp_tform():
    x = np.zeros((5, 5), dtype=np.double)
    x[2, 2] = 1
    theta = - np.pi / 2
    tform = SimilarityTransform(scale=1, rotation=theta, translation=(0, 4))

    x90 = warp(x, tform, order=1)
    assert_almost_equal(x90, np.rot90(x))

    x90 = warp(x, tform.inverse, order=1)
    assert_almost_equal(x90, np.rot90(x))


def test_warp_callable():
    x = np.zeros((5, 5), dtype=np.double)
    x[2, 2] = 1
    refx = np.zeros((5, 5), dtype=np.double)
    refx[1, 1] = 1

    def shift(xy):
        return xy + 1

    outx = warp(x, shift, order=1)
    assert_almost_equal(outx, refx)


@test_parallel()
def test_warp_matrix():
    x = np.zeros((5, 5), dtype=np.double)
    x[2, 2] = 1
    refx = np.zeros((5, 5), dtype=np.double)
    refx[1, 1] = 1

    matrix = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])

    # _warp_fast
    outx = warp(x, matrix, order=1)
    assert_almost_equal(outx, refx)
    # check for ndimage.map_coordinates
    outx = warp(x, matrix, order=5)


def test_warp_nd():
    for dim in range(2, 8):
        shape = dim * (5,)

        x = np.zeros(shape, dtype=np.double)
        x_c = dim * (2,)
        x[x_c] = 1
        refx = np.zeros(shape, dtype=np.double)
        refx_c = dim * (1,)
        refx[refx_c] = 1

        coord_grid = dim * (slice(0, 5, 1),)
        coords = np.array(np.mgrid[coord_grid]) + 1

        outx = warp(x, coords, order=0, cval=0)

        assert_almost_equal(outx, refx)


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
    assert_almost_equal(x90, np.rot90(x))


def test_const_cval_out_of_range():
    img = np.random.randn(100, 100)
    cval = - 10
    warped = warp(img, AffineTransform(translation=(10, 10)), cval=cval)
    assert np.sum(warped == cval) == (2 * 100 * 10 - 10 * 10)


def test_warp_identity():
    img = img_as_float(rgb2gray(data.astronaut()))
    assert len(img.shape) == 2
    assert np.allclose(img, warp(img, AffineTransform(rotation=0)))
    assert not np.allclose(img, warp(img, AffineTransform(rotation=0.1)))
    rgb_img = np.transpose(np.asarray([img, np.zeros_like(img), img]),
                           (1, 2, 0))
    warped_rgb_img = warp(rgb_img, AffineTransform(rotation=0.1))
    assert np.allclose(rgb_img, warp(rgb_img, AffineTransform(rotation=0)))
    assert not np.allclose(rgb_img, warped_rgb_img)
    # assert no cross-talk between bands
    assert np.all(0 == warped_rgb_img[:, :, 1])


def test_warp_coords_example():
    image = data.astronaut().astype(np.float32)
    assert 3 == image.shape[2]
    tform = SimilarityTransform(translation=(0, -10))
    coords = warp_coords(tform, (30, 30, 3))
    map_coordinates(image[:, :, 0], coords[:2])


def test_invalid():
    with testing.raises(ValueError):
        warp(np.ones((4, 3, 3, 3)),
             SimilarityTransform())


def test_inverse():
    tform = SimilarityTransform(scale=0.5, rotation=0.1)
    inverse_tform = SimilarityTransform(matrix=np.linalg.inv(tform.params))
    image = np.arange(10 * 10).reshape(10, 10).astype(np.double)
    assert_equal(warp(image, inverse_tform), warp(image, tform.inverse))


def test_slow_warp_nonint_oshape():
    image = np.random.rand(5, 5)

    with testing.raises(ValueError):
        warp(image, lambda xy: xy,
             output_shape=(13.1, 19.5))

    warp(image, lambda xy: xy, output_shape=(13.0001, 19.9999))


def test_zero_image_size():
    with testing.raises(ValueError):
        warp(np.zeros(0),
             SimilarityTransform())
    with testing.raises(ValueError):
        warp(np.zeros((0, 10)),
             SimilarityTransform())
    with testing.raises(ValueError):
        warp(np.zeros((10, 0)),
             SimilarityTransform())
    with testing.raises(ValueError):
        warp(np.zeros((10, 10, 0)),
             SimilarityTransform())


def test_linear_polar_mapping():
    output_coords = np.array([[0, 0],
                             [0, 90],
                             [0, 180],
                             [0, 270],
                             [99, 0],
                             [99, 180],
                             [99, 270],
                             [99, 45]])
    ground_truth = np.array([[100, 100],
                             [100, 100],
                             [100, 100],
                             [100, 100],
                             [199, 100],
                             [1, 100],
                             [100, 1],
                             [170.00357134, 170.00357134]])
    k_angle = 360 / (2 * np.pi)
    k_radius = 1
    center = (100, 100)
    coords = _linear_polar_mapping(output_coords, k_angle, k_radius, center)
    assert np.allclose(coords, ground_truth)


def test_log_polar_mapping():
    output_coords = np.array([[0, 0],
                              [0, 90],
                              [0, 180],
                              [0, 270],
                              [99, 0],
                              [99, 180],
                              [99, 270],
                              [99, 45]])
    ground_truth = np.array([[101, 100],
                             [100, 101],
                             [99, 100],
                             [100, 99],
                             [195.4992586, 100],
                             [4.5007414, 100],
                             [100, 4.5007414],
                             [167.52817336, 167.52817336]])
    k_angle = 360 / (2 * np.pi)
    k_radius = 100 / np.log(100)
    center = (100, 100)
    coords = _log_polar_mapping(output_coords, k_angle, k_radius, center)
    assert np.allclose(coords, ground_truth)
