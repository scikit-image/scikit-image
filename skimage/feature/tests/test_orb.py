import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from skimage.feature import keypoints_orb, descriptor_orb
from skimage.data import lena
from skimage.color import rgb2gray


def test_keypoints_orb_desired_no_of_keypoints():
    img = rgb2gray(lena())
    keypoints, orientations, scales = keypoints_orb(img, n_keypoints=10,
                                                    fast_n=12,
                                                    fast_threshold=0.20)
    exp_keypoints = np.array([[435, 180],
                              [436, 180],
                              [376, 156],
                              [455, 176],
                              [435, 180],
                              [269, 111],
                              [376, 156],
                              [311, 173],
                              [413,  70],
                              [311, 173]])
    exp_scales = np.array([0, 1, 0, 0, 2, 0, 1, 1, 0, 3])
    exp_orientations = np.array([-175.64733392, -167.94842949, -148.98350192,
                                 -142.03599837, -176.08535837,  -53.08162354,
                                 -150.89208271,   97.7693776 , -173.4479964 ,
                                 38.66312042])
    assert_array_equal(exp_keypoints, keypoints)
    assert_array_equal(exp_scales, scales)
    assert_almost_equal(exp_orientations, np.rad2deg(orientations))


def test_keypoints_orb_less_than_desired_no_of_keypoints():
    img = rgb2gray(lena())
    keypoints, orientations, scales = keypoints_orb(img, n_keypoints=15,
                                                    fast_n=12,
                                                    fast_threshold=0.33,
                                                    downscale=2, n_scales=2)
    exp_keypoints = np.array([[ 67, 157],
                              [247, 146],
                              [269, 111],
                              [413,  70],
                              [435, 180],
                              [230, 136],
                              [264, 336],
                              [330, 148],
                              [372, 156]])
    exp_scales = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    exp_orientations = np.array([-105.76503839,  -96.28973044,  -53.08162354,
                                 -173.4479964 , -175.64733392, -106.07927215,
                                 -163.40016243,   75.80865813, -154.73195911])
    assert_array_equal(exp_keypoints, keypoints)
    assert_array_equal(exp_scales, scales)
    assert_almost_equal(exp_orientations, np.rad2deg(orientations))


def test_descriptor_orb():
    img = rgb2gray(lena())
    keypoints, orientations, scales = keypoints_orb(img, n_keypoints=10,
                                                    fast_n=12,
                                                    fast_threshold=0.20)
    descriptors, filtered_keypoints = descriptor_orb(img, keypoints, orientations, scales)

    exp_filtered_keypoints = np.array([[435, 180],
                                       [376, 156],
                                       [455, 176],
                                       [269, 111],
                                       [413,  70],
                                       [436, 180],
                                       [376, 156],
                                       [311, 173],
                                       [435, 180],
                                       [311, 173]])

    descriptors_120_129 = np.array([[ True, False, False,  True, False, False, False, False, False, False],
                                    [ True,  True, False, False,  True, False, False,  True, False,  True],
                                    [False,  True,  True, False,  True, False,  True,  True,  True,  True],
                                    [False, False, False,  True,  True, False,  True, False,  True, False],
                                    [False,  True,  True,  True,  True, False,  True,  True,  True, False],
                                    [ True, False,  True,  True,  True, False, False, False,  True, False],
                                    [ True, False,  True, False,  True, False,  True,  True, False,  True],
                                    [ True,  True,  True,  True,  True,  True, False,  True,  True,  True],
                                    [ True,  True,  True, False,  True, False,  True,  True,  True, False],
                                    [ True,  True, False,  True,  True,  True, False,  True, False, True]],
                                    dtype=bool)

    assert_array_equal(exp_filtered_keypoints, filtered_keypoints)
    assert_array_equal(descriptors_120_129, descriptors[:, 120:130])


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
