from __future__ import print_function, division

import numpy as np
import math

from skimage.measure import hausdorff_distance, hausdorff_distance_region


def test_hausdorff_empty():
    empty = np.zeros((0, 2), dtype=np.float)
    non_empty = np.zeros((3, 2), dtype=np.float)
    assert math.isinf(hausdorff_distance(empty, non_empty))
    assert math.isinf(hausdorff_distance(non_empty, empty))
    assert hausdorff_distance(empty, empty) == 0.
