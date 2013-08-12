import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from skimage.data import moon
from skimage.feature import keypoints_censure


def test_keypoints_censure_color_image_unsupported_error():
    """Censure keypoints can be extracted from gray-scale images only."""
    img = np.zeros((20, 20, 3))
    assert_raises(ValueError, keypoints_censure, img)


def test_keypoints_censure_moon_image_dob():
    """Verify the actual Censure keypoints and their corresponding scale with
    the expected values for DoB filter."""
    img = moon()
    actual_kp_dob, actual_scale = keypoints_censure(img, 7, 'DoB', 0.15)
    expected_kp_dob = np.array([[ 21, 497],
                                [ 36,  46],
                                [119, 350],
                                [185, 177],
                                [287, 250],
                                [357, 239],
                                [463, 116],
                                [464, 132],
                                [467, 260]])
    expected_scale = np.array([3, 4, 4, 2, 2, 3, 2, 2, 2])

    assert_array_equal(expected_kp_dob, actual_kp_dob)
    assert_array_equal(expected_scale, actual_scale)


def test_keypoints_censure_moon_image_Octagon():
    """Verify the actual Censure keypoints and their corresponding scale with
    the expected values for Octagon filter."""
    img = moon()
    actual_kp_octagon, actual_scale = keypoints_censure(img, 7, 'Octagon',
                                                        0.15)
    expected_kp_octagon = np.array([[ 21, 496],
                                    [ 35,  46],
                                    [287, 250],
                                    [356, 239],
                                    [463, 116]])

    expected_scale = np.array([3, 4, 2, 2, 2])

    assert_array_equal(expected_kp_octagon, actual_kp_octagon)
    assert_array_equal(expected_scale, actual_scale)


def test_keypoints_censure_moon_image_STAR():
    """Verify the actual Censure keypoints and their corresponding scale with
    the expected values for STAR filter."""
    img = moon()
    actual_kp_star, actual_scale = keypoints_censure(img, 7, 'STAR', 0.15)
    expected_kp_star = np.array([[ 21, 497],
                                 [ 36,  46],
                                 [117, 356],
                                 [185, 177],
                                 [260, 227],
                                 [287, 250],
                                 [357, 239],
                                 [451, 281],
                                 [463, 116],
                                 [467, 260]])

    expected_scale = np.array([3, 3, 6, 2, 3, 2, 3, 5, 2, 2])

    assert_array_equal(expected_kp_star, actual_kp_star)
    assert_array_equal(expected_scale, actual_scale)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
