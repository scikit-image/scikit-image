from __future__ import print_function, division

import numpy as np
from numpy.testing import assert_almost_equal
import math
import itertools

from skimage.measure import hausdorff_distance, hausdorff_distance_region


def test_hausdorff_empty():
    empty = np.zeros((0, 2), dtype=np.float)
    non_empty = np.zeros((3, 2), dtype=np.float)
    assert math.isinf(hausdorff_distance(empty, non_empty))
    assert math.isinf(hausdorff_distance(non_empty, empty))
    assert hausdorff_distance(empty, empty) == 0.


def check_hausdorff_region_single(shape, coords_a, coords_b, distance):
    a = np.zeros(shape, dtype=np.bool)
    b = np.zeros(shape, dtype=np.bool)
    a[coords_a] = True
    b[coords_b] = True
    assert_almost_equal(hausdorff_distance_region(a, b), distance)


def test_hausdorff_region_single():
    shape = (5, 5)
    coords = [(0, 0), (3, 0), (1, 4), (4, 1)]
    for coords_a, coords_b in itertools.product(coords, repeat=2):
        distance = np.sqrt(sum((ca - cb)**2
                               for ca, cb in zip(coords_a, coords_b)))
        yield (check_hausdorff_region_single,
               shape, coords_a, coords_b, distance)
