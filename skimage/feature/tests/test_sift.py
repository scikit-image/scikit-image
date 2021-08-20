import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal

from skimage import data
from skimage._shared.testing import test_parallel
from skimage.feature import SIFT
from skimage.util.dtype import _convert

img = data.coins()


@test_parallel()
@pytest.mark.parametrize(
    'dtype', ['float32', 'float64', 'uint8', 'uint16', 'int64']
)
def test_keypoints_sift(dtype):
    _img = _convert(img, dtype)
    detector_extractor = SIFT()
    detector_extractor.detect_and_extract(_img)

    exp_keypoint_rows = np.array([18, 18, 25, 28, 30, 31, 31, 32, 32, 32])
    exp_keypoint_cols = np.array([331, 320, 310, 214, 204, 149, 323, 212, 338,
                                  208])

    exp_octaves = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    exp_position_rows = np.array([17.60995172, 18.24113663, 25.23896707,
                                  28.05011676, 29.75649063, 30.61094934,
                                  30.60819423, 31.56750745, 31.51146694,
                                  31.90489705])

    exp_position_cols = np.array([331.17101422, 320.00820411, 310.21729874,
                                  213.9275348, 204.46856377, 149.1101935,
                                  322.54098766, 212.24522397, 337.73737466,
                                  208.08643149])

    exp_orientations = np.array([0.26304959,  0.68454527,  0.98508569,
                                 0.07434552,  0.50036175, 0.34443952,
                                 1.47246218,  0.14766766, -1.81282453,
                                 0.42935128])

    exp_scales = np.array([2, 1, 3, 1, 2, 1, 1, 3, 1, 2])

    exp_sigmas = np.array([1.32408516, 0.93947056, 1.43985502, 0.98044104,
                           1.29242677, 1.00881748, 1.04676962, 1.55346773,
                           1.06313949, 1.29550204])

    exp_scalespace_sigmas = np.array([[0.8,  1.00793684,  1.26992084,  1.6,
                                       2.01587368, 2.53984168],
                                      [1.6,  2.01587368,  2.53984168,  3.2,
                                       4.03174736, 5.07968337],
                                      [3.2,  4.03174736,  5.07968337,  6.4,
                                       8.06349472, 10.15936673],
                                      [6.4,  8.06349472, 10.15936673, 12.8,
                                       16.12698944, 20.31873347],
                                      [12.8, 16.12698944, 20.31873347, 25.6,
                                       32.25397888, 40.63746693],
                                      [25.6, 32.25397888, 40.63746693, 51.2,
                                       64.50795775, 81.27493386]])

    assert_almost_equal(exp_keypoint_rows,
                        detector_extractor.keypoints[:10, 0])
    assert_almost_equal(exp_keypoint_cols,
                        detector_extractor.keypoints[:10, 1])
    assert_almost_equal(exp_octaves,
                        detector_extractor.octaves[:10])
    assert_almost_equal(exp_position_rows,
                        detector_extractor.positions[:10, 0], decimal=4)
    assert_almost_equal(exp_position_cols,
                        detector_extractor.positions[:10, 1], decimal=4)
    assert_almost_equal(exp_orientations,
                        detector_extractor.orientations[:10], decimal=4)
    assert_almost_equal(exp_scales,
                        detector_extractor.scales[:10])
    assert_almost_equal(exp_sigmas,
                        detector_extractor.sigmas[:10], decimal=4)
    assert_almost_equal(exp_scalespace_sigmas,
                        detector_extractor.scalespace_sigmas, decimal=4)

    detector_extractor2 = SIFT()
    detector_extractor2.detect(img)
    detector_extractor2.extract(img)
    assert_almost_equal(detector_extractor.keypoints[:10, 0],
                        detector_extractor2.keypoints[:10, 0])
    assert_almost_equal(detector_extractor.keypoints[:10, 0],
                        detector_extractor2.keypoints[:10, 0])


def test_descriptor_sift():
    detector_extractor = SIFT(n_hist=2, n_ori=4)
    exp_descriptors = np.array([[173,  29,  55,  32, 173,  16,  45,  80, 173,
                                 150, 172, 173, 173, 172,  68, 104],
                                [226,   9,  25,  25, 202,  25,  25,  29, 226,
                                 19, 170,  64, 226, 26, 70, 156],
                                [222,  17,  41,   9, 197,   5,  53,  36, 222,
                                 50, 148,  26, 222, 35, 135, 152],
                                [222,  25,  99,  16, 208,  31,  96,  18, 222,
                                 112,  28, 117, 222, 60, 29, 128],
                                [203,   9, 136,  44, 183,   9, 158,  35, 203,
                                 105,  71,  67, 203, 82, 28, 171],
                                [229,  45,  44,  22, 229,  15,  62,  97, 229,
                                 51,  71,  52, 229, 19, 61, 134],
                                [154,  53, 154,  50, 143, 154, 154,  77, 154,
                                 46, 154,  81, 154, 154, 154,  75],
                                [193,  75,  61,  37, 193,  27,  56,  71, 193,
                                 193,  29,  98, 193, 92, 21, 186],
                                [154,  82,  70, 138, 154,  89, 154,  77, 136,
                                 69, 154, 149, 154, 64, 154, 154],
                                [192,  38,  67,  39, 192,  24,  63, 106, 192,
                                 192,  54,  51, 192, 94, 36, 192]],
                               dtype=np.uint8
                               )

    detector_extractor.detect_and_extract(img)

    assert_equal(exp_descriptors, detector_extractor.descriptors[:10])

    keypoints_count = detector_extractor.keypoints.shape[0]
    assert keypoints_count == detector_extractor.descriptors.shape[0]
    assert keypoints_count == detector_extractor.orientations.shape[0]
    assert keypoints_count == detector_extractor.octaves.shape[0]
    assert keypoints_count == detector_extractor.positions.shape[0]
    assert keypoints_count == detector_extractor.scales.shape[0]
    assert keypoints_count == detector_extractor.scales.shape[0]


def test_no_descriptors_extracted_sift():
    img = np.ones((128, 128))
    detector_extractor = SIFT()
    with pytest.raises(RuntimeError):
        detector_extractor.detect_and_extract(img)
