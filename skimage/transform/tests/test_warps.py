import numpy as np
from scipy.ndimage import map_coordinates

from .._warps import (rescale, rotate, swirl, downscale_local_mean,
                      warp_polar, _linear_polar_mapping,
                      _log_polar_mapping)
from skimage import data, img_as_float
from skimage.color import rgb2gray
from skimage.draw import circle_perimeter_aa
from skimage.feature import peak_local_max
from skimage._shared import testing
from skimage._shared.testing import (assert_almost_equal, assert_equal,
                                     test_parallel)
from skimage._shared._warnings import expected_warnings


np.random.seed(0)


def test_warp_clip():
    x = np.zeros((5, 5), dtype=np.double)
    x[2, 2] = 1

    outx = rescale(x, 3, order=3, clip=False,
                   multichannel=False, anti_aliasing=False, mode='constant')
    assert outx.min() < 0

    outx = rescale(x, 3, order=3, clip=True,
                   multichannel=False, anti_aliasing=False, mode='constant')
    assert_almost_equal(outx.min(), 0)
    assert_almost_equal(outx.max(), 1)


def test_rotate():
    x = np.zeros((5, 5), dtype=np.double)
    x[1, 1] = 1
    x90 = rotate(x, 90)
    assert_almost_equal(x90, np.rot90(x))


def test_rotate_resize():
    x = np.zeros((10, 10), dtype=np.double)

    x45 = rotate(x, 45, resize=False)
    assert x45.shape == (10, 10)

    x45 = rotate(x, 45, resize=True)
    # new dimension should be d = sqrt(2 * (10/2)^2)
    assert x45.shape == (14, 14)


def test_rotate_center():
    x = np.zeros((10, 10), dtype=np.double)
    x[4, 4] = 1
    refx = np.zeros((10, 10), dtype=np.double)
    refx[2, 5] = 1
    x20 = rotate(x, 20, order=0, center=(0, 0))
    assert_almost_equal(x20, refx)
    x0 = rotate(x20, -20, order=0, center=(0, 0))
    assert_almost_equal(x0, x)


def test_rotate_resize_center():
    x = np.zeros((10, 10), dtype=np.double)
    x[0, 0] = 1

    ref_x45 = np.zeros((14, 14), dtype=np.double)
    ref_x45[6, 0] = 1
    ref_x45[7, 0] = 1

    x45 = rotate(x, 45, resize=True, center=(3, 3), order=0)
    # new dimension should be d = sqrt(2 * (10/2)^2)
    assert x45.shape == (14, 14)
    assert_equal(x45, ref_x45)


def test_rotate_resize_90():
    x90 = rotate(np.zeros((470, 230), dtype=np.double), 90, resize=True)
    assert x90.shape == (230, 470)


def test_rescale():
    # same scale factor
    x = np.zeros((5, 5), dtype=np.double)
    x[1, 1] = 1
    scaled = rescale(x, 2, order=0,
                     multichannel=False, anti_aliasing=False, mode='constant')
    ref = np.zeros((10, 10))
    ref[2:4, 2:4] = 1
    assert_almost_equal(scaled, ref)

    # different scale factors
    x = np.zeros((5, 5), dtype=np.double)
    x[1, 1] = 1

    scaled = rescale(x, (2, 1), order=0,
                     multichannel=False, anti_aliasing=False, mode='constant')
    ref = np.zeros((10, 5))
    ref[2:4, 1] = 1
    assert_almost_equal(scaled, ref)


def test_rescale_invalid_scale():
    x = np.zeros((10, 10, 3))
    with testing.raises(ValueError):
        rescale(x, (2, 2),
                multichannel=False, anti_aliasing=False, mode='constant')
    with testing.raises(ValueError):
        rescale(x, (2, 2, 2),
                multichannel=True, anti_aliasing=False, mode='constant')


def test_rescale_multichannel():
    # 1D + channels
    x = np.zeros((8, 3), dtype=np.double)
    scaled = rescale(x, 2, order=0, multichannel=True, anti_aliasing=False,
                     mode='constant')
    assert_equal(scaled.shape, (16, 3))
    # 2D
    scaled = rescale(x, 2, order=0, multichannel=False, anti_aliasing=False,
                     mode='constant')
    assert_equal(scaled.shape, (16, 6))

    # 2D + channels
    x = np.zeros((8, 8, 3), dtype=np.double)
    scaled = rescale(x, 2, order=0, multichannel=True, anti_aliasing=False,
                     mode='constant')
    assert_equal(scaled.shape, (16, 16, 3))
    # 3D
    scaled = rescale(x, 2, order=0, multichannel=False, anti_aliasing=False,
                     mode='constant')
    assert_equal(scaled.shape, (16, 16, 6))

    # 3D + channels
    x = np.zeros((8, 8, 8, 3), dtype=np.double)
    scaled = rescale(x, 2, order=0, multichannel=True, anti_aliasing=False,
                     mode='constant')
    assert_equal(scaled.shape, (16, 16, 16, 3))
    # 4D
    scaled = rescale(x, 2, order=0, multichannel=False, anti_aliasing=False,
                     mode='constant')
    assert_equal(scaled.shape, (16, 16, 16, 6))


def test_rescale_multichannel_multiscale():
    x = np.zeros((5, 5, 3), dtype=np.double)
    scaled = rescale(x, (2, 1), order=0, multichannel=True,
                     anti_aliasing=False, mode='constant')
    assert_equal(scaled.shape, (10, 5, 3))


def test_rescale_multichannel_defaults():
    x = np.zeros((8, 3), dtype=np.double)
    scaled = rescale(x, 2, order=0, anti_aliasing=False, mode='constant')
    assert_equal(scaled.shape, (16, 6))

    x = np.zeros((8, 8, 3), dtype=np.double)
    scaled = rescale(x, 2, order=0, anti_aliasing=False, mode='constant')
    assert_equal(scaled.shape, (16, 16, 6))


def test_swirl():
    image = img_as_float(data.checkerboard())

    swirl_params = {'radius': 80, 'rotation': 0, 'order': 2, 'mode': 'reflect'}

    with expected_warnings(['Bi-quadratic.*bug']):
        swirled = swirl(image, strength=10, **swirl_params)
        unswirled = swirl(swirled, strength=-10, **swirl_params)

    assert np.mean(np.abs(image - unswirled)) < 0.01

    swirl_params.pop('mode')

    with expected_warnings(['Bi-quadratic.*bug']):
        swirled = swirl(image, strength=10, **swirl_params)
        unswirled = swirl(swirled, strength=-10, **swirl_params)

    assert np.mean(np.abs(image[1:-1, 1:-1] - unswirled[1:-1, 1:-1])) < 0.01


def test_downscale():
    x = np.zeros((10, 10), dtype=np.double)
    x[2:4, 2:4] = 1
    scaled = rescale(x, 0.5, order=0, anti_aliasing=False,
                     multichannel=False, mode='constant')
    assert_equal(scaled.shape, (5, 5))
    assert_equal(scaled[1, 1], 1)
    assert_equal(scaled[2:, :].sum(), 0)
    assert_equal(scaled[:, 2:].sum(), 0)


def test_downscale_anti_aliasing():
    x = np.zeros((10, 10), dtype=np.double)
    x[2, 2] = 1
    scaled = rescale(x, 0.5, order=1, anti_aliasing=True,
                     multichannel=False, mode='constant')
    assert_equal(scaled.shape, (5, 5))
    assert np.all(scaled[:3, :3] > 0)
    assert_equal(scaled[3:, :].sum(), 0)
    assert_equal(scaled[:, 3:].sum(), 0)


def test_downscale_local_mean():
    image1 = np.arange(4 * 6).reshape(4, 6)
    out1 = downscale_local_mean(image1, (2, 3))
    expected1 = np.array([[4., 7.],
                          [16., 19.]])
    assert_equal(expected1, out1)

    image2 = np.arange(5 * 8).reshape(5, 8)
    out2 = downscale_local_mean(image2, (4, 5))
    expected2 = np.array([[14., 10.8],
                          [8.5, 5.7]])
    assert_equal(expected2, out2)


def test_keep_range():
    image = np.linspace(0, 2, 25).reshape(5, 5)
    out = rescale(image, 2, preserve_range=False, clip=True, order=0,
                  mode='constant', multichannel=False, anti_aliasing=False)
    assert out.min() == 0
    assert out.max() == 2

    out = rescale(image, 2, preserve_range=True, clip=True, order=0,
                  mode='constant', multichannel=False, anti_aliasing=False)
    assert out.min() == 0
    assert out.max() == 2

    out = rescale(image.astype(np.uint8), 2, preserve_range=False,
                  mode='constant', multichannel=False, anti_aliasing=False,
                  clip=True, order=0)
    assert out.min() == 0
    assert out.max() == 2 / 255.0


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


def test_linear_warp_polar():
    radii = [5, 10, 15, 20]
    image = np.zeros([51, 51])
    for rad in radii:
        rr, cc, val = circle_perimeter_aa(25, 25, rad)
        image[rr, cc] = val
    warped = warp_polar(image, radius=25)
    profile = warped.mean(axis=0)
    peaks = peak_local_max(profile)
    assert np.alltrue([peak in radii for peak in peaks])


def test_log_warp_polar():
    radii = [np.exp(2), np.exp(3), np.exp(4), np.exp(5),
             np.exp(5)-1, np.exp(5)+1]
    radii = [int(x) for x in radii]
    image = np.zeros([301, 301])
    for rad in radii:
        rr, cc, val = circle_perimeter_aa(150, 150, rad)
        image[rr, cc] = val
    warped = warp_polar(image, radius=200, scaling='log')
    profile = warped.mean(axis=0)
    peaks = peak_local_max(profile)
    gaps = peaks[:-1]-peaks[1:]
    assert np.alltrue([x >= 38 and x <= 40 for x in gaps])


def test_invalid_scaling_polar():
    with testing.raises(ValueError):
        warp_polar(np.zeros((10, 10)), (5, 5), scaling='invalid')
    with testing.raises(ValueError):
        warp_polar(np.zeros((10, 10)), (5, 5), scaling=None)


def test_invalid_dimensions_polar():
    with testing.raises(ValueError):
        warp_polar(np.zeros((10, 10, 3)), (5, 5))
    with testing.raises(ValueError):
        warp_polar(np.zeros((10, 10)), (5, 5), multichannel=True)
    with testing.raises(ValueError):
        warp_polar(np.zeros((10, 10, 10, 3)), (5, 5), multichannel=True)
