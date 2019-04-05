import numpy as np
from scipy.ndimage import map_coordinates

from skimage.transform._warps import _stackcopy
from skimage.transform import (warp_coords, resize, rescale,
                               downscale_local_mean)

from skimage.transform_xy import SimilarityTransform

from skimage import data

from skimage._shared import testing
from skimage._shared.testing import assert_almost_equal, assert_equal
from skimage._shared._warnings import expected_warnings


np.random.seed(0)


def test_stackcopy():
    layers = 4
    x = np.empty((3, 3, layers))
    y = np.eye(3, 3)
    _stackcopy(x, y)
    for i in range(layers):
        assert_almost_equal(x[..., i], y)


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
    # ensure multichannel=None matches the previous default behaviour

    # 2D: multichannel should default to False
    x = np.zeros((8, 3), dtype=np.double)
    with expected_warnings(['multichannel']):
        scaled = rescale(x, 2, order=0, anti_aliasing=False, mode='constant')
    assert_equal(scaled.shape, (16, 6))

    # 3D: multichannel should default to True
    x = np.zeros((8, 8, 3), dtype=np.double)
    with expected_warnings(['multichannel']):
        scaled = rescale(x, 2, order=0, anti_aliasing=False, mode='constant')
    assert_equal(scaled.shape, (16, 16, 3))


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


def test_resize2d():
    x = np.zeros((5, 5), dtype=np.double)
    x[1, 1] = 1
    resized = resize(x, (10, 10), order=0, anti_aliasing=False,
                     mode='constant')
    ref = np.zeros((10, 10))
    ref[2:4, 2:4] = 1
    assert_almost_equal(resized, ref)


def test_resize3d_keep():
    # keep 3rd dimension
    x = np.zeros((5, 5, 3), dtype=np.double)
    x[1, 1, :] = 1
    resized = resize(x, (10, 10), order=0, anti_aliasing=False,
                     mode='constant')
    with testing.raises(ValueError):
        # output_shape too short
        resize(x, (10, ), order=0, anti_aliasing=False, mode='constant')
    ref = np.zeros((10, 10, 3))
    ref[2:4, 2:4, :] = 1
    assert_almost_equal(resized, ref)
    resized = resize(x, (10, 10, 3), order=0, anti_aliasing=False,
                     mode='constant')
    assert_almost_equal(resized, ref)


def test_resize3d_resize():
    # resize 3rd dimension
    x = np.zeros((5, 5, 3), dtype=np.double)
    x[1, 1, :] = 1
    resized = resize(x, (10, 10, 1), order=0, anti_aliasing=False,
                     mode='constant')
    ref = np.zeros((10, 10, 1))
    ref[2:4, 2:4] = 1
    assert_almost_equal(resized, ref)


def test_resize3d_2din_3dout():
    # 3D output with 2D input
    x = np.zeros((5, 5), dtype=np.double)
    x[1, 1] = 1
    resized = resize(x, (10, 10, 1), order=0, anti_aliasing=False,
                     mode='constant')
    ref = np.zeros((10, 10, 1))
    ref[2:4, 2:4] = 1
    assert_almost_equal(resized, ref)


def test_resize2d_4d():
    # resize with extra output dimensions
    x = np.zeros((5, 5), dtype=np.double)
    x[1, 1] = 1
    out_shape = (10, 10, 1, 1)
    resized = resize(x, out_shape, order=0, anti_aliasing=False,
                     mode='constant')
    ref = np.zeros(out_shape)
    ref[2:4, 2:4, ...] = 1
    assert_almost_equal(resized, ref)


def test_resize_nd():
    for dim in range(1, 6):
        shape = 2 + np.arange(dim) * 2
        x = np.ones(shape)
        out_shape = np.asarray(shape) * 1.5
        resized = resize(x, out_shape, order=0, mode='reflect',
                         anti_aliasing=False)
        expected_shape = 1.5 * shape
        assert_equal(resized.shape, expected_shape)
        assert np.all(resized == 1)


def test_resize3d_bilinear():
    # bilinear 3rd dimension
    x = np.zeros((5, 5, 2), dtype=np.double)
    x[1, 1, 0] = 0
    x[1, 1, 1] = 1
    resized = resize(x, (10, 10, 1), order=1, mode='constant',
                     anti_aliasing=False)
    ref = np.zeros((10, 10, 1))
    ref[1:5, 1:5, :] = 0.03125
    ref[1:5, 2:4, :] = 0.09375
    ref[2:4, 1:5, :] = 0.09375
    ref[2:4, 2:4, :] = 0.28125
    assert_almost_equal(resized, ref)


def test_warp_coords_example():
    image = data.astronaut().astype(np.float32)
    assert 3 == image.shape[2]
    tform = SimilarityTransform(translation=(0, -10))
    coords = warp_coords(tform, (30, 30, 3))
    map_coordinates(image[:, :, 0], coords[:2])


def test_downsize():
    x = np.zeros((10, 10), dtype=np.double)
    x[2:4, 2:4] = 1
    scaled = resize(x, (5, 5), order=0, anti_aliasing=False, mode='constant')
    assert_equal(scaled.shape, (5, 5))
    assert_equal(scaled[1, 1], 1)
    assert_equal(scaled[2:, :].sum(), 0)
    assert_equal(scaled[:, 2:].sum(), 0)


def test_downsize_anti_aliasing():
    x = np.zeros((10, 10), dtype=np.double)
    x[2, 2] = 1
    scaled = resize(x, (5, 5), order=1, anti_aliasing=True, mode='constant')
    assert_equal(scaled.shape, (5, 5))
    assert np.all(scaled[:3, :3] > 0)
    assert_equal(scaled[3:, :].sum(), 0)
    assert_equal(scaled[:, 3:].sum(), 0)

    sigma = 0.125
    out_size = (5, 5)
    resize(x, out_size, order=1, mode='constant',
           anti_aliasing=True, anti_aliasing_sigma=sigma)
    resize(x, out_size, order=1, mode='edge',
           anti_aliasing=True, anti_aliasing_sigma=sigma)
    resize(x, out_size, order=1, mode='symmetric',
           anti_aliasing=True, anti_aliasing_sigma=sigma)
    resize(x, out_size, order=1, mode='reflect',
           anti_aliasing=True, anti_aliasing_sigma=sigma)
    resize(x, out_size, order=1, mode='wrap',
           anti_aliasing=True, anti_aliasing_sigma=sigma)

    with testing.raises(ValueError):  # Unknown mode, or cannot translate mode
        resize(x, out_size, order=1, mode='non-existent',
               anti_aliasing=True, anti_aliasing_sigma=sigma)


def test_downsize_anti_aliasing_invalid_stddev():
    x = np.zeros((10, 10), dtype=np.double)
    with testing.raises(ValueError):
        resize(x, (5, 5), order=0, anti_aliasing=True, anti_aliasing_sigma=-1,
               mode='constant')
    with expected_warnings(["Anti-aliasing standard deviation greater"]):
        resize(x, (5, 15), order=0, anti_aliasing=True,
               anti_aliasing_sigma=(1, 1), mode="reflect")
        resize(x, (5, 15), order=0, anti_aliasing=True,
               anti_aliasing_sigma=(0, 1), mode="reflect")


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
                  mode='constant', multichannel='False', anti_aliasing=False)
    assert out.min() == 0
    assert out.max() == 2

    out = rescale(image, 2, preserve_range=True, clip=True, order=0,
                  mode='constant', multichannel='False', anti_aliasing=False)
    assert out.min() == 0
    assert out.max() == 2

    out = rescale(image.astype(np.uint8), 2, preserve_range=False,
                  mode='constant', multichannel='False', anti_aliasing=False,
                  clip=True, order=0)
    assert out.min() == 0
    assert out.max() == 2 / 255.0
