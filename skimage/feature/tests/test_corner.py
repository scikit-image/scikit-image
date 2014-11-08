import numpy as np
from numpy.testing import (assert_array_equal, assert_raises,
                           assert_almost_equal)

from skimage import data
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.morphology import octagon

from skimage.feature import (corner_moravec, corner_harris, corner_shi_tomasi,
                             corner_subpix, peak_local_max, corner_peaks,
                             corner_kitchen_rosenfeld, corner_foerstner,
                             corner_fast, corner_orientations,
                             structure_tensor, structure_tensor_eigvals,
                             hessian_matrix, hessian_matrix_eigvals,
                             hessian_matrix_det)


def test_structure_tensor():
    square = np.zeros((5, 5))
    square[2, 2] = 1
    Axx, Axy, Ayy = structure_tensor(square, sigma=0.1)
    assert_array_equal(Axx, np.array([[ 0,  0,  0,  0,  0],
                                      [ 0,  1,  0,  1,  0],
                                      [ 0,  4,  0,  4,  0],
                                      [ 0,  1,  0,  1,  0],
                                      [ 0,  0,  0,  0,  0]]))
    assert_array_equal(Axy, np.array([[ 0,  0,  0,  0,  0],
                                      [ 0,  1,  0, -1,  0],
                                      [ 0,  0,  0, -0,  0],
                                      [ 0, -1, -0,  1,  0],
                                      [ 0,  0,  0,  0,  0]]))
    assert_array_equal(Ayy, np.array([[ 0,  0,  0,  0,  0],
                                      [ 0,  1,  4,  1,  0],
                                      [ 0,  0,  0,  0,  0],
                                      [ 0,  1,  4,  1,  0],
                                      [ 0,  0,  0,  0,  0]]))


def test_hessian_matrix():
    square = np.zeros((5, 5))
    square[2, 2] = 1
    Hxx, Hxy, Hyy = hessian_matrix(square, sigma=0.1)
    assert_array_equal(Hxx, np.array([[0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0]]))
    assert_array_equal(Hxy, np.array([[0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0]]))
    assert_array_equal(Hyy, np.array([[0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0]]))


def test_structure_tensor_eigvals():
    square = np.zeros((5, 5))
    square[2, 2] = 1
    Axx, Axy, Ayy = structure_tensor(square, sigma=0.1)
    l1, l2 = structure_tensor_eigvals(Axx, Axy, Ayy)
    assert_array_equal(l1, np.array([[0, 0, 0, 0, 0],
                                     [0, 2, 4, 2, 0],
                                     [0, 4, 0, 4, 0],
                                     [0, 2, 4, 2, 0],
                                     [0, 0, 0, 0, 0]]))
    assert_array_equal(l2, np.array([[0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0]]))


def test_hessian_matrix_eigvals():
    square = np.zeros((5, 5))
    square[2, 2] = 1
    Hxx, Hxy, Hyy = hessian_matrix(square, sigma=0.1)
    l1, l2 = hessian_matrix_eigvals(Hxx, Hxy, Hyy)
    assert_array_equal(l1, np.array([[0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0]]))
    assert_array_equal(l2, np.array([[0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0]]))


def test_hessian_matrix_det():
    image = np.zeros((5, 5))
    image[2, 2] = 1
    det = hessian_matrix_det(image, 5)
    assert_almost_equal(det, 0, decimal = 3)


def test_square_image():
    im = np.zeros((50, 50)).astype(float)
    im[:25, :25] = 1.

    # Moravec
    results = peak_local_max(corner_moravec(im))
    # interest points along edge
    assert len(results) == 57

    # Harris
    results = peak_local_max(corner_harris(im, method='k'))
    # interest at corner
    assert len(results) == 1

    results = peak_local_max(corner_harris(im, method='eps'))
    # interest at corner
    assert len(results) == 1

    # Shi-Tomasi
    results = peak_local_max(corner_shi_tomasi(im))
    # interest at corner
    assert len(results) == 1


def test_noisy_square_image():
    im = np.zeros((50, 50)).astype(float)
    im[:25, :25] = 1.
    np.random.seed(seed=1234)
    im = im + np.random.uniform(size=im.shape) * .2

    # Moravec
    results = peak_local_max(corner_moravec(im))
    # undefined number of interest points
    assert results.any()

    # Harris
    results = peak_local_max(corner_harris(im, sigma=1.5, method='k'))
    assert len(results) == 1
    results = peak_local_max(corner_harris(im, sigma=1.5, method='eps'))
    assert len(results) == 1

    # Shi-Tomasi
    results = peak_local_max(corner_shi_tomasi(im, sigma=1.5))
    assert len(results) == 1


def test_squared_dot():
    im = np.zeros((50, 50))
    im[4:8, 4:8] = 1
    im = img_as_float(im)

    # Moravec fails

    # Harris
    results = peak_local_max(corner_harris(im))
    assert (results == np.array([[6, 6]])).all()

    # Shi-Tomasi
    results = peak_local_max(corner_shi_tomasi(im))
    assert (results == np.array([[6, 6]])).all()


def test_rotated_img():
    """
    The harris filter should yield the same results with an image and it's
    rotation.
    """
    im = img_as_float(data.astronaut().mean(axis=2))
    im_rotated = im.T

    # Moravec
    results = peak_local_max(corner_moravec(im))
    results_rotated = peak_local_max(corner_moravec(im_rotated))
    assert (np.sort(results[:, 0]) == np.sort(results_rotated[:, 1])).all()
    assert (np.sort(results[:, 1]) == np.sort(results_rotated[:, 0])).all()

    # Harris
    results = peak_local_max(corner_harris(im))
    results_rotated = peak_local_max(corner_harris(im_rotated))
    assert (np.sort(results[:, 0]) == np.sort(results_rotated[:, 1])).all()
    assert (np.sort(results[:, 1]) == np.sort(results_rotated[:, 0])).all()

    # Shi-Tomasi
    results = peak_local_max(corner_shi_tomasi(im))
    results_rotated = peak_local_max(corner_shi_tomasi(im_rotated))
    assert (np.sort(results[:, 0]) == np.sort(results_rotated[:, 1])).all()
    assert (np.sort(results[:, 1]) == np.sort(results_rotated[:, 0])).all()


def test_subpix_edge():
    img = np.zeros((50, 50))
    img[:25, :25] = 255
    img[25:, 25:] = 255
    corner = peak_local_max(corner_harris(img), num_peaks=1)
    subpix = corner_subpix(img, corner)
    assert_array_equal(subpix[0], (24.5, 24.5))


def test_subpix_dot():
    img = np.zeros((50, 50))
    img[25, 25] = 255
    corner = peak_local_max(corner_harris(img), num_peaks=1)
    subpix = corner_subpix(img, corner)
    assert_array_equal(subpix[0], (25, 25))


def test_subpix_no_class():
    img = np.zeros((50, 50))
    subpix = corner_subpix(img, np.array([[25, 25]]))
    assert_array_equal(subpix[0], (np.nan, np.nan))

    img[25, 25] = 1e-10
    corner = peak_local_max(corner_harris(img), num_peaks=1)
    subpix = corner_subpix(img, np.array([[25, 25]]))
    assert_array_equal(subpix[0], (np.nan, np.nan))


def test_subpix_border():
    img = np.zeros((50, 50))
    img[1:25,1:25] = 255
    img[25:-1,25:-1] = 255
    corner = corner_peaks(corner_harris(img), min_distance=1)
    subpix = corner_subpix(img, corner, window_size=11)
    ref = np.array([[ 0.52040816,  0.52040816],
                    [ 0.52040816, 24.47959184],
                    [24.47959184,  0.52040816],
                    [24.5       , 24.5       ],
                    [24.52040816, 48.47959184],
                    [48.47959184, 24.52040816],
                    [48.47959184, 48.47959184]])
    assert_almost_equal(subpix, ref)


def test_num_peaks():
    """For a bunch of different values of num_peaks, check that
    peak_local_max returns exactly the right amount of peaks. Test
    is run on the astronaut image in order to produce a sufficient number of corners"""

    img_corners = corner_harris(rgb2gray(data.astronaut()))

    for i in range(20):
        n = np.random.random_integers(20)
        results = peak_local_max(img_corners, num_peaks=n)
        assert (results.shape[0] == n)


def test_corner_peaks():
    response = np.zeros((5, 5))
    response[2:4, 2:4] = 1

    corners = corner_peaks(response, exclude_border=False)
    assert len(corners) == 1

    corners = corner_peaks(response, exclude_border=False, min_distance=0)
    assert len(corners) == 4

    corners = corner_peaks(response, exclude_border=False, min_distance=0,
                           indices=False)
    assert np.sum(corners) == 4


def test_blank_image_nans():
    """Some of the corner detectors had a weakness in terms of returning
    NaN when presented with regions of constant intensity. This should
    be fixed by now. We test whether each detector returns something
    finite in the case of constant input"""

    detectors = [corner_moravec, corner_harris, corner_shi_tomasi,
                 corner_kitchen_rosenfeld, corner_foerstner]
    constant_image = np.zeros((20, 20))

    for det in detectors:
        response = det(constant_image)
        assert np.all(np.isfinite(response))


def test_corner_fast_image_unsupported_error():
    img = np.zeros((20, 20, 3))
    assert_raises(ValueError, corner_fast, img)


def test_corner_fast_lena():
    img = rgb2gray(data.astronaut())
    expected = np.array([[101, 198],
                        [140, 205],
                        [141, 242],
                        [177, 156],
                        [188, 113],
                        [197, 148],
                        [213, 117],
                        [223, 375],
                        [232, 266],
                        [245, 137],
                        [249, 171],
                        [300, 244],
                        [305,  57],
                        [325, 245],
                        [339, 242],
                        [346, 279],
                        [353, 172],
                        [358, 307],
                        [362, 252],
                        [362, 328],
                        [363, 192],
                        [364, 147],
                        [369, 159],
                        [374, 171],
                        [379, 183],
                        [387, 195],
                        [390, 149],
                        [401, 197],
                        [403, 162],
                        [413, 181],
                        [444, 310],
                        [464, 251],
                        [476, 250],
                        [489, 155],
                        [492, 139],
                        [494, 169],
                        [496, 266]])
    actual = corner_peaks(corner_fast(img, 12, 0.3))
    assert_array_equal(actual, expected)


def test_corner_orientations_image_unsupported_error():
    img = np.zeros((20, 20, 3))
    assert_raises(ValueError, corner_orientations, img,
                  np.asarray([[7, 7]]), np.ones((3, 3)))


def test_corner_orientations_even_shape_error():
    img = np.zeros((20, 20))
    assert_raises(ValueError, corner_orientations, img,
                  np.asarray([[7, 7]]), np.ones((4, 4)))


def test_corner_orientations_lena():
    img = rgb2gray(data.lena())
    corners = corner_peaks(corner_fast(img, 11, 0.35))
    expected = np.array([-1.9195897 , -3.03159624, -1.05991162, -2.89573739,
                         -2.61607644, 2.98660159])
    actual = corner_orientations(img, corners, octagon(3, 2))
    assert_almost_equal(actual, expected)


def test_corner_orientations_square():
    square = np.zeros((12, 12))
    square[3:9, 3:9] = 1
    corners = corner_peaks(corner_fast(square, 9), min_distance=1)
    actual_orientations = corner_orientations(square, corners, octagon(3, 2))
    actual_orientations_degrees = np.rad2deg(actual_orientations)
    expected_orientations_degree = np.array([  45.,  135.,  -45., -135.])
    assert_array_equal(actual_orientations_degrees,
                       expected_orientations_degree)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
