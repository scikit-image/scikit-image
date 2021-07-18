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

    exp_keypoint_rows = np.array([18, 20, 24, 30, 32, 32, 32, 32, 34, 34])
    exp_keypoint_cols = np.array([331, 318, 346, 205, 338, 208, 213, 326, 227,
                                  331])

    exp_octaves = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    exp_position_rows = np.array([17.87767327, 20.17989539, 24.45986247,
                                  30.02442796, 31.70717422, 32.11748417,
                                  31.74324821, 32.18909856, 33.51604029,
                                  33.69935759])

    exp_position_cols = np.array([331.40911199, 317.82389875, 346.4354115,
                                  204.73282333, 337.85377173, 208.37761189,
                                  212.50188137, 325.91408422, 226.66768553,
                                  331.01485921])

    exp_orientations = np.array([0.27204716, 0.68527923, -0.62994593,
                                 0.49786301, 1.66494918, 0.41734936,
                                 0.14866038, 1.44832243, -0.39065165,
                                 2.38010732])

    exp_scales = np.array([2, 1, 2, 2, 1, 2, 3, 1, 2, 1])

    exp_sigmas = np.array([1.30662976, 0.92791825, 1.30417738, 1.27288812,
                           1.05923417, 1.28155738, 1.50214994, 0.95579367,
                           1.35887452, 0.98095441])

    exp_scalespace_sigmas = np.array([[0.8, 1.00793684, 1.26992084, 1.6,
                                       2.01587368, 2.53984168],
                                      [1.6, 2.01587368, 2.53984168, 3.2,
                                       4.03174736, 5.07968337],
                                      [3.2, 4.03174736, 5.07968337, 6.4,
                                       8.06349472, 10.15936673],
                                      [6.4, 8.06349472, 10.15936673, 12.8,
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
    exp_descriptors = np.array(
        [
            [172, 28, 57, 35, 172, 16, 46, 82, 172, 150, 172, 172, 172, 171,
             71, 105],
            [225, 8, 25, 24, 225, 14, 26, 22, 225, 20, 186, 46, 225, 17, 109,
             79],
            [200, 125, 60, 64, 200, 29, 138, 69, 108, 31, 73, 198, 137, 114,
             145, 149],
            [202, 9, 135, 44, 184, 9, 155, 35, 202, 106, 75, 70, 202, 86, 31,
             169],
            [151, 117, 151, 109, 151, 151, 99, 120, 151, 125, 151, 56, 63, 151,
             151, 77],
            [191, 39, 68, 38, 191, 25, 63, 103, 191, 191, 58, 53, 191, 101, 38,
             191],
            [194, 72, 59, 36, 194, 26, 54, 72, 194, 194, 28, 97, 194, 89, 21,
             187],
            [156, 20, 156, 50, 156, 156, 156, 58, 142, 77, 156, 96, 156, 116,
             156, 105],
            [198, 69, 83, 18, 198, 31, 74, 103, 198, 175, 40, 93, 198, 107, 37,
             147],
            [137, 137, 137, 137, 129, 137, 137, 109, 137, 91, 53, 137, 137,
             132, 124, 137]
        ],
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
