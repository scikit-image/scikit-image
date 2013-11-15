import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from skimage.feature import keypoints_orb, descriptor_orb
from skimage.data import lena
from skimage.color import rgb2gray


def test_keypoints_orb_desired_no_of_keypoints():
    img = rgb2gray(lena())
    keypoints = keypoints_orb(img, n_keypoints=10, fast_n=12,
                              fast_threshold=0.20)
    exp_row = np.array([ 435.  ,  435.6 ,  376.  ,  455.  ,  434.88,  269.  ,
                         375.6 ,  310.8 ,  413.  ,  311.04])
    exp_col = np.array([ 180. ,  180. ,  156. ,  176. ,  180. ,  111. ,
                         156. ,  172.8,   70. ,  172.8])

    exp_octaves = np.array([ 1.   ,  1.2  ,  1.   ,  1.   ,  1.44 ,  1.   ,
                             1.2  ,  1.2  ,  1.   ,  1.728])

    exp_orientations = np.array([-175.64733392, -167.94842949, -148.98350192,
                                 -142.03599837, -176.08535837,  -53.08162354,
                                 -150.89208271,   97.7693776 , -173.4479964 ,
                                 38.66312042])
    exp_response = np.array([ 0.96770745,  0.81027306,  0.72376257,
                              0.5626413 ,  0.5097993 ,  0.44351774,
                              0.39154173,  0.39084861,  0.39063076,
                              0.37602487])
    assert_almost_equal(exp_row, keypoints.row)
    assert_almost_equal(exp_col, keypoints.col)
    assert_almost_equal(exp_octaves, keypoints.octave)
    assert_almost_equal(exp_response, keypoints.response)
    assert_almost_equal(exp_orientations, np.rad2deg(keypoints.orientation))


def test_keypoints_orb_less_than_desired_no_of_keypoints():
    img = rgb2gray(lena())
    keypoints = keypoints_orb(img, n_keypoints=15, fast_n=12,
                              fast_threshold=0.33, downscale=2, n_scales=2)

    exp_row = np.array([  67.,  247.,  269.,  413.,  435.,  230.,  264.,
                         330.,  372.])
    exp_col = np.array([ 157.,  146.,  111.,   70.,  180.,  136.,  336.,
                         148.,  156.])

    exp_octaves = np.array([ 1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,  2.])

    exp_orientations = np.array([-105.76503839,  -96.28973044,  -53.08162354,
                                 -173.4479964 , -175.64733392, -106.07927215,
                                 -163.40016243,   75.80865813, -154.73195911])

    exp_response = np.array([ 0.13197835,  0.24931321,  0.44351774,
                              0.39063076,  0.96770745,  0.04935129,
                              0.21431068,  0.15826555,  0.42403573])

    assert_almost_equal(exp_row, keypoints.row)
    assert_almost_equal(exp_col, keypoints.col)
    assert_almost_equal(exp_octaves, keypoints.octave)
    assert_almost_equal(exp_response, keypoints.response)
    assert_almost_equal(exp_orientations, np.rad2deg(keypoints.orientation))


def test_descriptor_orb():
    img = rgb2gray(lena())
    keypoints = keypoints_orb(img, n_keypoints=10, fast_n=12,
                              fast_threshold=0.20)
    descriptors, filtered_keypoints = descriptor_orb(img, keypoints)

    descriptors_120_129 = np.array([[ True, False, False,  True, False, False, False, False, False, False],
                                    [ True,  True, False, False,  True, False, False,  True, False,  True],
                                    [False,  True,  True, False,  True, False,  True,  True,  True,  True],
                                    [False, False, False,  True,  True, False,  True, False,  True, False],
                                    [False,  True,  True,  True,  True, False,  True,  True,  True, False],
                                    [ True, False,  True,  True,  True, False, False, False,  True, False],
                                    [ True, False,  True, False,  True, False,  True,  True, False,  True],
                                    [ True,  True,  True,  True,  True,  True, False,  True,  True,  True],
                                    [ True,  True,  True, False,  True, False,  True,  True,  True, False],
                                    [ True, False, False, False, False, False,  True,  True,  True, False]],
                                    dtype=bool)


    assert_array_equal(descriptors_120_129, descriptors[:, 120:130])


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
