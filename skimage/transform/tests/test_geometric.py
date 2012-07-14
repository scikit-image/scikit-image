import numpy as np
from numpy.testing import assert_array_almost_equal

from skimage.transform.geometric import _stackcopy
from skimage.transform import estimate_transformation, homography, warp, \
    fast_homography, SimilarityTransformation, AffineTransformation, \
    ProjectiveTransformation, PolynomialTransformation
from skimage import transform as tf, data, img_as_float
from skimage.color import rgb2gray


SRC = np.array([
    [-12.3705, -10.5075],
    [-10.7865, 15.4305],
    [8.6985, 10.8675],
    [11.4975, -9.5715],
    [7.8435, 7.4835],
    [-5.3325, 6.5025],
    [6.7905, -6.3765],
    [-6.1695, -0.8235],
])
DST = np.array([
    [0, 0],
    [0, 5800],
    [4900, 5800],
    [4900, 0],
    [4479, 4580],
    [1176, 3660],
    [3754, 790],
    [1024, 1931],
])


def test_stackcopy():
    layers = 4
    x = np.empty((3, 3, layers))
    y = np.eye(3, 3)
    _stackcopy(x, y)
    for i in range(layers):
        assert_array_almost_equal(x[..., i], y)


def test_similarity_estimation():
    #: exact solution
    tform = estimate_transformation('similarity', SRC[:2, :], DST[:2, :])
    assert_array_almost_equal(tform.forward(SRC[:2, :]), DST[:2, :])
    assert_array_almost_equal(tform.reverse(tform.forward(SRC)), SRC)

    #: over-determined
    tform = estimate_transformation('similarity', SRC, DST)
    ref = np.array(
        [[2.3632898110e+02, -5.5876792257e+00, 2.5331569391e+03],
         [5.5876792257e+00, 2.3632898110e+02, 2.4358232635e+03],
         [0.0000000000e+00, 0.0000000000e+00, 1.0000000000e+00]])
    assert_array_almost_equal(tform.matrix, ref)
    assert_array_almost_equal(tform.reverse(tform.forward(SRC)), SRC)


def test_similarity_explicit():
    tform = SimilarityTransformation()
    scale = 0.1
    rotation = 1
    translation = (1, 1)
    tform.from_params(scale, rotation, translation)
    assert_array_almost_equal(tform.scale, scale)
    assert_array_almost_equal(tform.rotation, rotation)
    assert_array_almost_equal(tform.translation, translation)


def test_affine_estimation():
    #: exact solution
    tform = estimate_transformation('affine', SRC[:3, :], DST[:3, :])
    assert_array_almost_equal(tform.forward(SRC[:3, :]), DST[:3, :])
    assert_array_almost_equal(tform.reverse(tform.forward(SRC)), SRC)

    #: over-determined
    tform = estimate_transformation('affine', SRC, DST)
    ref = np.array(
        [[2.2573930047e+02, 7.1588596765e+00, 2.5126622012e+03],
         [2.1234856855e+01, 2.4931019555e+02, 2.4143862183e+03],
         [0.0000000000e+00, 0.0000000000e+00, 1.0000000000e+00]])
    assert_array_almost_equal(tform.matrix, ref)
    assert_array_almost_equal(tform.reverse(tform.forward(SRC)), SRC)


def test_affine_explicit():
    tform = AffineTransformation()
    scale = (0.1, 0.13)
    rotation = 1
    shear = 0.1
    translation = (1, 1)
    tform.from_params(scale, rotation, shear, translation)
    assert_array_almost_equal(tform.scale, scale)
    assert_array_almost_equal(tform.rotation, rotation)
    assert_array_almost_equal(tform.shear, shear)
    assert_array_almost_equal(tform.translation, translation)


def test_projective():
    #: exact solution
    tform = estimate_transformation('projective', SRC[:4, :], DST[:4, :])
    ref = np.array(
        [[  1.9466901291e+02, -1.1888183994e+01, 2.2832379309e+03],
         [ -8.6910077540e+00,  2.2162069773e+02, 2.2211673699e+03],
         [ -1.2695966735e-02, -9.6053624285e-03, 1.0000000000e+00]])
    assert_array_almost_equal(tform.matrix, ref, 6)
    assert_array_almost_equal(tform.reverse(tform.forward(SRC)), SRC)

    #: over-determined
    tform = estimate_transformation('projective', SRC[:4, :], DST[:4, :])
    ref = np.array(
        [[  1.9466901291e+02, -1.1888183994e+01, 2.2832379309e+03],
         [ -8.6910077540e+00,  2.2162069773e+02, 2.2211673699e+03],
         [ -1.2695966735e-02, -9.6053624285e-03, 1.0000000000e+00]])
    assert_array_almost_equal(tform.matrix, ref, 6)
    assert_array_almost_equal(tform.reverse(tform.forward(SRC)), SRC)


def test_polynomial():
    tform = estimate_transformation('polynomial', SRC, DST, order=10)
    assert_array_almost_equal(tform.forward(SRC), DST, 6)


def test_union():
    tform1 = SimilarityTransformation()
    scale1 = 0.1
    rotation1 = 1
    translation1 = (0, 0)
    tform1.from_params(scale1, rotation1, translation1)

    tform2 = SimilarityTransformation()
    scale2 = 0.1
    rotation2 = 1
    translation2 = (0, 0)
    tform2.from_params(scale2, rotation2, translation2)

    tform = tform1 + tform2
    tform = tform1 * tform2

    assert_array_almost_equal(tform.scale, scale1 * scale2)
    assert_array_almost_equal(tform.rotation, rotation1 + rotation2)


def test_warp():
    x = np.zeros((5, 5), dtype=np.uint8)
    x[2, 2] = 255
    x = img_as_float(x)
    theta = -np.pi/2
    tform = SimilarityTransformation()
    tform.from_params(1, theta, (0, 4))

    x90 = warp(x, tform, order=1)
    assert_array_almost_equal(x90, np.rot90(x))

    x90 = warp(x, tform.reverse, order=1)
    assert_array_almost_equal(x90, np.rot90(x))


def test_homography():
    x = np.zeros((5, 5), dtype=np.uint8)
    x[1, 1] = 255
    x = img_as_float(x)
    theta = -np.pi/2
    M = np.array([[np.cos(theta),-np.sin(theta),0],
                  [np.sin(theta), np.cos(theta),4],
                  [0,             0,            1]])
    x90 = homography(x, M, order=1)
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

    for mode in ('constant', 'mirror', 'wrap'):
        p0 = homography(img, H, mode=mode, order=1)
        p1 = fast_homography(img, H, mode=mode)
        p1 = np.round(p1)

        ## import matplotlib.pyplot as plt
        ## f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
        ## ax0.imshow(img)
        ## ax1.imshow(p0, cmap=plt.cm.gray)
        ## ax2.imshow(p1, cmap=plt.cm.gray)
        ## ax3.imshow(np.abs(p0 - p1), cmap=plt.cm.gray)
        ## plt.show()

        d = np.mean(np.abs(p0 - p1))
        assert d < 0.2


def test_swirl():
    image = img_as_float(data.checkerboard())

    swirl_params = {'radius': 80, 'rotation': 0, 'order': 2, 'mode': 'reflect'}
    swirled = tf.swirl(image, strength=10, **swirl_params)
    unswirled = tf.swirl(swirled, strength=-10, **swirl_params)

    assert np.mean(np.abs(image - unswirled)) < 0.01


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
