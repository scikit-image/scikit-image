from __future__ import print_function, division

import numpy as np
from numpy.testing import assert_almost_equal
import itertools
import pytest

from skimage._shared.testing import parametrize
from skimage.metrics import hausdorff_distance


def test_hausdorff_empty():
    empty = np.zeros((0, 2), dtype=np.bool)
    non_empty = np.zeros((3, 2), dtype=np.bool)
    with pytest.raises(ValueError):
        hausdorff_distance(empty, non_empty)
    with pytest.raises(ValueError):
        hausdorff_distance(non_empty, empty)
    assert hausdorff_distance(empty, empty) == 0.


def test_hausdorff_simple():
    points_a = (3, 0)
    points_b = (6, 0)
    shape = (7, 1)
    coords_a = np.zeros(shape, dtype=np.bool)
    coords_b = np.zeros(shape, dtype=np.bool)
    coords_a[points_a] = True
    coords_b[points_b] = True
    distance = np.sqrt(sum((ca - cb) ** 2
                           for ca, cb in zip(points_a, points_b)))
    assert_almost_equal(hausdorff_distance(coords_a, coords_b), distance)


class TestHausdorffRegionSingle:
    points = [(0, 0), (3, 0), (1, 4), (4, 1)]

    def check_hausdorff_region_single(self, points_a, points_b):
        shape = (5, 5)
        coords_a = np.zeros(shape, dtype=np.bool)
        coords_b = np.zeros(shape, dtype=np.bool)
        coords_a[points_a] = True
        coords_b[points_b] = True

        distance = np.sqrt(sum((ca - cb) ** 2
                               for ca, cb in zip(points_a, points_b)))
        assert_almost_equal(hausdorff_distance(coords_a, coords_b), distance)

    @parametrize("points_a, points_b", itertools.product(points, repeat=2))
    def test_hausdorff_region_single(self, points_a, points_b):
        self.check_hausdorff_region_single(points_a, points_b)


class TestHausdorffRegionDifferentPoints:
    points_a = [(5, 4), (4, 5), (3, 4), (4, 3)]
    points_b = [(6, 4), (2, 6), (2, 4), (4, 0)]

    def check_hausdorff_region_different_points(self, points_a, points_b):
        shape = (7, 7)
        coords_a = np.zeros(shape, dtype=np.bool)
        coords_b = np.zeros(shape, dtype=np.bool)
        coords_a[points_a] = True
        coords_b[points_b] = True

        distance = np.sqrt(sum((ca - cb) ** 2
                               for ca, cb in zip(points_a, points_b)))
        assert_almost_equal(hausdorff_distance(coords_a, coords_b), distance)

    @parametrize("points_a, points_b", itertools.product(points_a, points_b))
    def test_hausdorff_region_different_points(self, points_a, points_b):
        self.check_hausdorff_region_different_points(points_a, points_b)


def test_gallery():
    shape = (60, 60)

    # Create a diamond-like shape where the four corners form the 1st set
    # of points
    x_diamond = 30
    y_diamond = 30
    r = 10

    plt_x = [0, 1, 0, -1]
    plt_y = [1, 0, -1, 0]

    set_ax = [(x_diamond + r * x) for x in plt_x]
    set_ay = [(y_diamond + r * y) for y in plt_y]

    # Create a kite-like shape where the four corners form the 2nd set of
    # points
    x_kite = 30
    y_kite = 30
    x_r = 15
    y_r = 20

    set_bx = [(x_kite + x_r * x) for x in plt_x]
    set_by = [(y_kite + y_r * y) for y in plt_y]

    # Set up the data to compute the hausdorff distance
    coords_a = np.zeros(shape, dtype=np.bool)
    coords_b = np.zeros(shape, dtype=np.bool)

    for x, y in zip(set_ax, set_ay):
        coords_a[(x, y)] = True

    for x, y in zip(set_bx, set_by):
        coords_b[(x, y)] = True

    # Test the hausdorff function on the coordinates
    # Should return 10, the distance between the furthest tip of the kite and
    # its closest point on the diamond, which is the furthest someone can make
    # you travel to encounter your nearest neighboring point on the other set.
    assert_almost_equal(hausdorff_distance(coords_a, coords_b), 10.)


class Test3DHausdorffRegion:
    points_a_3d = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    points_b_3d = [(0, 0, 2), (0, 2, 0), (2, 0, 0)]
    hausdorff_distances_list = []

    def check_3d_hausdorff_region(self, points_a, points_b):
        shape = (3, 3, 3)
        coords_a = np.zeros(shape, dtype=np.bool)
        coords_b = np.zeros(shape, dtype=np.bool)
        coords_a[points_a] = True
        coords_b[points_b] = True

        distance = np.sqrt(sum((ca - cb) ** 2
                               for ca, cb in zip(points_a, points_b)))
        hausdorff_distance_3d = hausdorff_distance(coords_a, coords_b)
        assert_almost_equal(hausdorff_distance_3d, distance)
        self.hausdorff_distances_list.append(hausdorff_distance_3d)

    @parametrize("points_a_3d, points_b_3d",
                 itertools.product(points_a_3d, points_b_3d))
    def test_3d_hausdorff_region(self, points_a_3d, points_b_3d):
        self.check_3d_hausdorff_region(points_a_3d, points_b_3d)

    def test_range_of_hausdorff_distances(self):
        assert_almost_equal(min(self.hausdorff_distances_list), 1)
        assert_almost_equal(max(self.hausdorff_distances_list), np.sqrt(5))
