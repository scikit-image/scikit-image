import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from skimage.data import moon
from skimage.feature import censure_keypoints


def test_censure_keypoints_color_image_unsupported_error():
    """Censure keypoints can be extracted from gray-scale images only."""
    img = np.zeros((20, 20, 3))
    assert_raises(ValueError, censure_keypoints, img)


def test_censure_keypoints_moon_image_DoB():
    """Verify the actual Censure keypoints and their corresponding scale with
    the expected values for DoB filter."""
    img = moon()
    actual_kp_DoB, actual_scale = censure_keypoints(img, 7, 'DoB', 0.15)
    expected_kp_DoB = np.array([[  4, 507],
                                [  8, 503],
                                [ 12, 499],
                                [ 21, 497],
                                [ 36,  46],
                                [119, 350],
                                [185, 177],
                                [287, 250],
                                [357, 239],
                                [463, 116],
                                [464, 132],
                                [467, 260]])
    expected_scale = np.array([2, 4, 6, 3, 4, 4, 2, 2, 3, 2, 2, 2])

    assert_array_equal(expected_kp_DoB, actual_kp_DoB)
    assert_array_equal(expected_scale, actual_scale)


def test_censure_keypoints_moon_image_Octagon():
    """Verify the actual Censure keypoints and their corresponding scale with
    the expected values for Octagon filter."""
    img = moon()
    actual_kp_Octagon, actual_scale = censure_keypoints(img, 7, 'Octagon', 0.15)
    expected_kp_Octagon = np.array([[287, 250],
                                    [356, 239],
                                    [463, 116],
                                    [ 21, 496],
                                    [ 35,  46]])

    expected_scale = np.array([2, 2, 2, 3, 4], dtype=np.int32)

    assert_array_equal(expected_kp_Octagon, actual_kp_Octagon)
    assert_array_equal(expected_scale, actual_scale)


def test_censure_keypoints_moon_image_STAR():
    """Verify the actual Censure keypoints and their corresponding scale with
    the expected values for STAR filter."""
    img = moon()
    actual_kp_STAR, actual_scale = censure_keypoints(img, 7, 'STAR', 0.15)
    expected_kp_STAR = np.array([[185, 177],
                                 [287, 250],
                                 [463, 116],
                                 [467, 260],
                                 [ 21, 497],
                                 [ 36,  46],
                                 [260, 227],
                                 [357, 239],
                                 [451, 281],
                                 [117, 356]])
    expected_scale = np.array([2, 2, 2, 2, 3, 3, 3, 3, 5, 6], dtype=np.int32)

    assert_array_equal(expected_kp_STAR, actual_kp_STAR)
    assert_array_equal(expected_scale, actual_scale)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
