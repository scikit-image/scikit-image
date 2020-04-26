import numpy as np
from scipy.ndimage import map_coordinates

from skimage.data import checkerboard, astronaut
from skimage.util.dtype import img_as_float
from skimage.color.colorconv import rgb2gray
from skimage.draw.draw import circle_perimeter_aa
from skimage.feature.peak import peak_local_max
from skimage._shared import testing
from skimage._shared.testing import (assert_almost_equal, assert_equal,
                                     test_parallel)
from skimage._shared._warnings import expected_warnings

from skimage.transform._warps import (_stackcopy,
                                      _linear_polar_mapping,
                                      _log_polar_mapping, warp,
                                      warp_coords, rotate, resize,
                                      rescale, warp_polar, swirl,
                                      downscale_local_mean)
from skimage.transform._geometric import (AffineTransform,
                                          ProjectiveTransform,
                                          SimilarityTransform)


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


def test_resize_dtype():
    x = np.zeros((5, 5))
    x_f32 = x.astype(np.float32)
    x_u8 = x.astype(np.uint8)
    x_b = x.astype(bool)

    assert resize(x, (10, 10), preserve_range=False).dtype == x.dtype
    assert resize(x, (10, 10), preserve_range=True).dtype == x.dtype
    assert resize(x_u8, (10, 10), preserve_range=False).dtype == np.double
    assert resize(x_u8, (10, 10), preserve_range=True).dtype == np.double
    assert resize(x_b, (10, 10), preserve_range=False).dtype == np.double
    assert resize(x_b, (10, 10), preserve_range=True).dtype == np.double
    assert resize(x_f32, (10, 10), preserve_range=False).dtype == x_f32.dtype
    assert resize(x_f32, (10, 10), preserve_range=True).dtype == x_f32.dtype


def test_swirl():
    image = img_as_float(checkerboard())

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


def test_const_cval_out_of_range():
    img = np.random.randn(100, 100)
    cval = - 10
    warped = warp(img, AffineTransform(translation=(10, 10)), cval=cval)
    assert np.sum(warped == cval) == (2 * 100 * 10 - 10 * 10)


def test_warp_identity():
    img = img_as_float(rgb2gray(astronaut()))
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
    image = astronaut().astype(np.float32)
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
    peaks_coord = peak_local_max(profile)
    peaks_coord.sort(axis=0)
    gaps = peaks_coord[1:] - peaks_coord[:-1]
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


def test_bool_img_rescale():
    img = np.ones((12, 18), dtype=bool)
    img[2:-2, 4:-4] = False
    res = rescale(img, 0.5)

    expected = np.ones((6, 9))
    expected[1:-1, 2:-2] = False

    assert_equal(res, expected)


def test_bool_img_resize():
    img = np.ones((12, 18), dtype=bool)
    img[2:-2, 4:-4] = False
    res = resize(img, (6, 9))

    expected = np.ones((6, 9))
    expected[1:-1, 2:-2] = False

    assert_equal(res, expected)


def test_boll_array_warnings():
    img = np.zeros((10, 10), dtype=bool)

    with expected_warnings(['Input image dtype is bool']):
        rescale(img, 0.5, anti_aliasing=True)

    with expected_warnings(['Input image dtype is bool']):
        resize(img, (5, 5), anti_aliasing=True)

    with expected_warnings(['Input image dtype is bool']):
        rescale(img, 0.5, order=1)

    with expected_warnings(['Input image dtype is bool']):
        resize(img, (5, 5), order=1)

    with expected_warnings(['Input image dtype is bool']):
        warp(img, np.eye(3), order=1)
