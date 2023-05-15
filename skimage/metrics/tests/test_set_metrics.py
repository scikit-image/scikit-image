import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.spatial import distance

from skimage._shared._warnings import expected_warnings
from skimage.metrics import (hausdorff_distance, hausdorff_pair,
                                hausdorff_distance_mask, hausdorff_pair_mask)
from skimage.morphology import disk, erosion


def test_hausdorff_empty():
    gt = np.zeros((3, 3), dtype=bool)
    pred = np.zeros((3, 3), dtype=bool)
    assert hausdorff_distance(gt, pred) == 0.0  # standard Hausdorff
    assert (
        hausdorff_distance(gt, pred, method="modified") == 0.0
    )  # modified Hausdorff
    with expected_warnings(["One or both of the images is empty"]):
        assert_array_equal(hausdorff_pair(gt, pred), [(), ()])
    assert hausdorff_distance_mask(gt, pred) == 0.0
    with expected_warnings(["One or both of the images is empty"]):
        assert_array_equal(hausdorff_pair_mask(gt, pred), [(), ()])


def test_hausdorff_shape_mismatch():
    gt = np.zeros((3, 3), dtype=bool)
    pred = np.zeros((4, 4), dtype=bool)
    with pytest.raises(ValueError):
        hausdorff_distance(gt, pred)


def test_hausdorff_one_empty():
    gt = np.zeros((3, 3), dtype=bool)
    pred = np.zeros((3, 3), dtype=bool)
    pred[1, 1] = True
    assert hausdorff_distance(gt, pred) == np.inf
    with expected_warnings(["One or both of the images is empty"]):
        assert_array_equal(hausdorff_pair(gt, pred), [(), ()])
    assert hausdorff_distance_mask(gt, pred) == np.inf
    with expected_warnings(["One or both of the images is empty"]):
        assert_array_equal(hausdorff_pair_mask(gt, pred), [(), ()])


def test_hausdorff_simple():
    points_a = (3, 0)
    points_b = (6, 0)
    shape = (7, 1)
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)
    coords_a[points_a] = True
    coords_b[points_b] = True

    d = distance.cdist([points_a], [points_b])
    dist = max(np.max(np.min(d, axis=0)), np.max(np.min(d, axis=1)))
    dist_modified = max(np.mean(np.min(d, axis=0)), np.mean(np.min(d, axis=1)))
    assert_almost_equal(hausdorff_distance(coords_a, coords_b), dist)
    assert_array_equal(hausdorff_pair(coords_a, coords_b), (points_a, points_b))
    assert_almost_equal(
        hausdorff_distance(
            coords_a,
            coords_b,
            method="modified",
        ),
        dist_modified,
    )


def test_hausdorff_mask():
    ground_truth = np.zeros((10, 10), dtype=bool)
    predicted = ground_truth.copy()
    ground_truth[2:9, 2:9] = True
    predicted[4:7, 2:9] = True
    dist = hausdorff_distance_mask(ground_truth, predicted)
    assert_almost_equal(dist, 2.0)
    p0, p1 = hausdorff_pair_mask(ground_truth, predicted)
    assert distance.euclidean(p0, p1) == dist


@pytest.mark.parametrize("points_a", [(0, 0), (3, 0), (1, 4), (4, 1)])
@pytest.mark.parametrize("points_b", [(0, 0), (3, 0), (1, 4), (4, 1)])
def test_hausdorff_region_single(points_a, points_b):
    shape = (5, 5)
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)
    coords_a[points_a] = True
    coords_b[points_b] = True

    d = distance.cdist([points_a], [points_b])
    assert_almost_equal(hausdorff_distance(coords_a, coords_b), d)
    assert_almost_equal(
        hausdorff_distance(coords_a, coords_b, method="modified"), d
    )


@pytest.mark.parametrize("points_a", [(5, 4), (4, 5), (3, 4), (4, 3)])
@pytest.mark.parametrize("points_b", [(6, 4), (2, 6), (2, 4), (4, 0)])
def test_hausdorff_region_different_points(points_a, points_b):
    shape = (7, 7)
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)
    coords_a[points_a] = True
    coords_b[points_b] = True

    d = distance.cdist([points_a], [points_b])
    dist = max(np.max(np.min(d, axis=0)), np.max(np.min(d, axis=1)))
    dist_modified = max(np.mean(np.min(d, axis=0)), np.mean(np.min(d, axis=1)))

    assert_almost_equal(hausdorff_distance(coords_a, coords_b), dist)
    assert_array_equal(hausdorff_pair(coords_a, coords_b), (points_a, points_b))
    assert_almost_equal(
        hausdorff_distance(coords_a, coords_b, method="modified"), dist_modified
    )


def test_gallery():
    # Creates a "ground truth" binary mask with a disk, and a partially overlapping "predicted" rectangle
    ground_truth = np.zeros((100, 100), dtype=bool)
    predicted = ground_truth.copy()

    ground_truth[30:71, 30:71] = disk(20)
    predicted[25:65, 40:70] = True

    # Creates "contours" image by xor-ing an erosion
    se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    gt_contour = ground_truth ^ erosion(ground_truth, se)
    predicted_contour = predicted ^ erosion(predicted, se)

    # From the "contours image":
    # Computes & display the distance & the corresponding pair of points
    distance = hausdorff_distance(gt_contour, predicted_contour)
    pair = hausdorff_pair(gt_contour, predicted_contour)

    assert_almost_equal(distance, np.sqrt(sum((ca - cb) ** 2 for ca, cb in zip(pair[0], pair[1]))))

    # From the segmentation masks directly:
    # Computes & display the distance & the corresponding pair of points
    distance = hausdorff_distance_mask(ground_truth, predicted)
    pair = hausdorff_pair_mask(ground_truth, predicted)

    assert_almost_equal(distance, np.sqrt(sum((ca - cb) ** 2 for ca, cb in zip(pair[0], pair[1]))))


@pytest.mark.parametrize("points_a", [(0, 0, 1), (0, 1, 0), (1, 0, 0)])
@pytest.mark.parametrize("points_b", [(0, 0, 2), (0, 2, 0), (2, 0, 0)])
def test_3d_hausdorff_region(points_a, points_b):
    shape = (3, 3, 3)
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)
    coords_a[points_a] = True
    coords_b[points_b] = True

    dist = np.sqrt(sum((ca - cb) ** 2 for ca, cb in zip(points_a, points_b)))
    d = distance.cdist([points_a], [points_b])
    dist_modified = max(np.mean(np.min(d, axis=0)), np.mean(np.min(d, axis=1)))
    assert_almost_equal(hausdorff_distance(coords_a, coords_b), dist)
    assert_array_equal(hausdorff_pair(coords_a, coords_b), (points_a, points_b))
    assert_almost_equal(
        hausdorff_distance(coords_a, coords_b, method="modified"), dist_modified
    )
