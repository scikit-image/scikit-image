import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, run_module_suite
from skimage.feature import ORB
from skimage import data
from skimage.color import rgb2gray


img = rgb2gray(data.lena())


def test_keypoints_orb_desired_no_of_keypoints():
    detector_extractor = ORB(n_keypoints=10, fast_n=12, fast_threshold=0.20)
    detector_extractor.detect(img)

    exp_rows = np.array([ 435.  ,  435.6 ,  376.  ,  455.  ,  434.88,  269.  ,
                          375.6 ,  310.8 ,  413.  ,  311.04])
    exp_cols = np.array([ 180. ,  180. ,  156. ,  176. ,  180. ,  111. ,
                          156. ,  172.8,   70. ,  172.8])

    exp_scales = np.array([ 1.   ,  1.2  ,  1.   ,  1.   ,  1.44 ,  1.   ,
                            1.2  ,  1.2  ,  1.   ,  1.728])

    exp_orientations = np.array([-175.64733392, -167.94842949, -148.98350192,
                                 -142.03599837, -176.08535837,  -53.08162354,
                                 -150.89208271,   97.7693776 , -173.4479964 ,
                                 38.66312042])
    exp_response = np.array([ 0.96770745,  0.81027306,  0.72376257,
                              0.5626413 ,  0.5097993 ,  0.44351774,
                              0.39154173,  0.39084861,  0.39063076,
                              0.37602487])

    assert_almost_equal(exp_rows, detector_extractor.keypoints[:, 0])
    assert_almost_equal(exp_cols, detector_extractor.keypoints[:, 1])
    assert_almost_equal(exp_scales, detector_extractor.scales)
    assert_almost_equal(exp_response, detector_extractor.responses)
    assert_almost_equal(exp_orientations,
                        np.rad2deg(detector_extractor.orientations), 5)

    detector_extractor.detect_and_extract(img)
    assert_almost_equal(exp_rows, detector_extractor.keypoints[:, 0])
    assert_almost_equal(exp_cols, detector_extractor.keypoints[:, 1])


def test_keypoints_orb_less_than_desired_no_of_keypoints():
    detector_extractor = ORB(n_keypoints=15, fast_n=12,
                             fast_threshold=0.33, downscale=2, n_scales=2)
    detector_extractor.detect(img)

    exp_rows = np.array([  67.,  247.,  269.,  413.,  435.,  230.,  264.,
                          330.,  372.])
    exp_cols = np.array([ 157.,  146.,  111.,   70.,  180.,  136.,  336.,
                          148.,  156.])

    exp_scales = np.array([ 1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,  2.])

    exp_orientations = np.array([-105.76503839,  -96.28973044,  -53.08162354,
                                 -173.4479964 , -175.64733392, -106.07927215,
                                 -163.40016243,   75.80865813, -154.73195911])

    exp_response = np.array([ 0.13197835,  0.24931321,  0.44351774,
                              0.39063076,  0.96770745,  0.04935129,
                              0.21431068,  0.15826555,  0.42403573])

    assert_almost_equal(exp_rows, detector_extractor.keypoints[:, 0])
    assert_almost_equal(exp_cols, detector_extractor.keypoints[:, 1])
    assert_almost_equal(exp_scales, detector_extractor.scales)
    assert_almost_equal(exp_response, detector_extractor.responses)
    assert_almost_equal(exp_orientations,
                        np.rad2deg(detector_extractor.orientations), 5)

    detector_extractor.detect_and_extract(img)
    assert_almost_equal(exp_rows, detector_extractor.keypoints[:, 0])
    assert_almost_equal(exp_cols, detector_extractor.keypoints[:, 1])


def test_descriptor_orb():
    detector_extractor = ORB(fast_n=12, fast_threshold=0.20)

    exp_descriptors = np.array([[ True, False,  True,  True, False, False, False, False, False, False],
                                [False, False,  True,  True, False,  True,  True, False,  True,  True],
                                [ True, False, False, False,  True, False,  True,  True,  True, False],
                                [ True, False, False,  True, False,  True,  True, False, False, False],
                                [False,  True,  True,  True, False, False, False,  True,  True, False],
                                [False, False, False, False, False,  True, False,  True,  True,  True],
                                [False,  True,  True,  True,  True, False, False,  True, False,  True],
                                [ True,  True,  True, False,  True,  True,  True,  True, False, False],
                                [ True,  True, False,  True,  True,  True,  True, False, False, False],
                                [ True, False, False, False, False,  True, False, False,  True,  True],
                                [ True, False, False, False,  True,  True,  True, False, False, False],
                                [False, False,  True, False,  True, False, False,  True, False, False],
                                [False, False,  True,  True, False, False, False, False, False,  True],
                                [ True,  True, False, False, False,  True,  True,  True,  True,  True],
                                [ True,  True,  True, False, False,  True, False,  True,  True, False],
                                [False,  True,  True, False, False,  True,  True,  True,  True,  True],
                                [ True,  True,  True, False, False, False, False,  True,  True,  True],
                                [False, False, False, False,  True, False, False,  True,  True, False],
                                [False,  True, False, False,  True, False, False, False,  True,  True],
                                [ True, False,  True, False, False, False,  True,  True, False, False]], dtype=bool)

    detector_extractor.detect(img)
    detector_extractor.extract(img, detector_extractor.keypoints,
                               detector_extractor.scales,
                               detector_extractor.orientations)
    assert_equal(exp_descriptors,
                 detector_extractor.descriptors[100:120, 10:20])

    detector_extractor.detect_and_extract(img)
    assert_equal(exp_descriptors,
                 detector_extractor.descriptors[100:120, 10:20])


if __name__ == '__main__':
    run_module_suite()
