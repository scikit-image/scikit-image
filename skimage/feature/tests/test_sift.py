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
    detector_extractor.detect(_img)

    exp_keypoint_rows = np.array([201, 41, 209, 180, 142, 182, 208, 34, 43,
                                  64])
    exp_keypoint_cols = np.array([264, 328, 271, 268, 326, 227, 223, 331, 345,
                                  289])

    exp_octaves = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    exp_position_rows = np.array([201.41658469, 41.06040806, 208.58110252,
                                  179.54736107, 141.87835206, 182.33889284,
                                  207.93740719, 33.69935734, 42.7844453,
                                  63.93211704])
    exp_position_cols = np.array([263.95690148, 327.82541304, 271.1789404,
                                  268.01588646, 326.07482427, 227.2984818,
                                  223.25775302, 331.01485918, 344.66424442,
                                  288.85184003])

    exp_orientations = np.array([-1.28679603, -1.98182586, 1.87902015,
                                 -2.61004661, 2.87064235, -1.66924637,
                                 0.55223721, 2.38010737, 1.89140887,
                                 -1.96767703])

    exp_scales = np.array([2, 1, 2, 3, 1, 1, 1, 1, 1, 2])

    exp_sigmas = np.array([1.32392846, 0.98984512, 1.25943425, 1.49632906,
                           0.90311007, 0.95008988, 0.93648624, 0.98095435,
                           1.0230875, 1.21606771])

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

    assert detector_extractor.keypoints.dtype == np.int64
    assert detector_extractor.octaves.dtype == np.int64
    assert detector_extractor.positions.dtype == np.float64
    assert detector_extractor.orientations.dtype == np.float64
    assert detector_extractor.scales.dtype == np.int64
    assert detector_extractor.sigmas.dtype == np.float64
    assert detector_extractor.scalespace_sigmas.dtype == np.float64

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
                        detector_extractor.orientations[:10])
    assert_almost_equal(exp_scales,
                        detector_extractor.scales[:10])
    assert_almost_equal(exp_sigmas,
                        detector_extractor.sigmas[:10], decimal=4)
    assert_almost_equal(exp_scalespace_sigmas,
                        detector_extractor.scalespace_sigmas)

    detector_extractor2 = SIFT()
    detector_extractor2.detect(img)
    detector_extractor2.extract(img)
    assert_almost_equal(detector_extractor.keypoints[:10, 0],
                        detector_extractor2.keypoints[:10, 0])
    assert_almost_equal(detector_extractor.keypoints[:10, 0],
                        detector_extractor2.keypoints[:10, 0])


def test_descriptor_sift():
    detector_extractor = SIFT(n_hist=2, n_ori=4)
    exp_descriptors = np.array([[130, 69, 124, 148, 148, 142, 148, 61, 148, 76,
                                 148, 119, 148, 57, 148, 148],
                                [142, 135, 142, 135, 142, 131, 142, 112, 142,
                                 76, 142, 98, 142, 119, 120, 92],
                                [157, 157, 68, 35, 157, 105, 157, 107, 157,
                                 157, 93, 95, 157, 52, 152, 125],
                                [121, 60, 176, 67, 132, 71, 153, 48, 176, 168,
                                 176, 42, 176, 54, 176, 44],
                                [218, 22, 8, 42, 218, 42, 15, 82, 218, 25, 20,
                                 102, 218, 33, 5, 218],
                                [161, 161, 61, 59, 161, 127, 98, 118, 161, 161,
                                 71, 54, 161, 124, 121, 141],
                                [133, 71, 74, 106, 134, 95, 98, 96, 167, 173,
                                 173, 80, 170, 81, 173, 123],
                                [137, 137, 137, 137, 129, 137, 137, 109, 137,
                                 91, 53, 137, 137, 132, 124, 137],
                                [118, 38, 167, 123, 136, 65, 167, 107, 167, 59,
                                 167, 103, 167, 54, 167, 111],
                                [169, 62, 98, 138, 169, 42, 46, 169, 169, 103,
                                 127, 119, 169, 67, 79, 169]],
                               dtype=np.uint8)

    detector_extractor.detect_and_extract(img)

    assert_equal(exp_descriptors, detector_extractor.descriptors[0:10])

    keypoints_count = detector_extractor.keypoints.shape[0]
    assert keypoints_count == detector_extractor.descriptors.shape[0]
    assert keypoints_count == detector_extractor.orientations.shape[0]
    assert keypoints_count == detector_extractor.octaves.shape[0]
    assert keypoints_count == detector_extractor.positions.shape[0]
    assert keypoints_count == detector_extractor.scales.shape[0]
    assert keypoints_count == detector_extractor.scales.shape[0]


def test_no_descriptors_extracted_orb():
    img = np.ones((128, 128))
    detector_extractor = SIFT()
    with pytest.raises(RuntimeError):
        detector_extractor.detect_and_extract(img)
