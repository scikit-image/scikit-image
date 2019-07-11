from __future__ import print_function, division

import numpy as np
from numpy.testing import assert_almost_equal
import math
import itertools

from skimage._shared.testing import parametrize
from skimage.measure import hausdorff_distance, hausdorff_distance_region


def test_hausdorff_empty():
    empty = np.zeros((0, 2), dtype=np.float)
    non_empty = np.zeros((3, 2), dtype=np.float)
    assert math.isinf(hausdorff_distance(empty, non_empty))
    assert math.isinf(hausdorff_distance(non_empty, empty))
    assert hausdorff_distance(empty, empty) == 0.


points = [(0, 0), (3, 0), (1, 4), (4, 1)]
@parametrize("points_a, points_b", itertools.product(points, repeat=2))
def test_hausdorff_region_single(points_a, points_b):
    check_hausdorff_region_single(points_a, points_b)


def check_hausdorff_region_single(points_a, points_b):
    shape = (5, 5)
    coords_a = np.zeros(shape, dtype=np.bool)
    coords_b = np.zeros(shape, dtype=np.bool)
    coords_a[points_a] = True
    coords_b[points_b] = True

    distance = np.sqrt(sum((ca - cb) ** 2
                           for ca, cb in zip(points_a, points_b)))
    assert_almost_equal(hausdorff_distance_region(coords_a, coords_b), distance)
