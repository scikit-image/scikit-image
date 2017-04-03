import math
import unittest

import numpy as np

from skimage.morphology import criteria
from scipy import ndimage as ndi
import pdb

eps = 1e-12


def diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    t = ((a - b)**2).sum()
    return math.sqrt(t)


class TestExtrema(unittest.TestCase):

    def test_area_closing(self):
        "test for area closing"
        data = np.array(
            [[250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
             [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
             [250, 250, 100, 100, 250, 250, 250,  50, 250, 250],
             [250, 250, 100, 100, 250, 250, 250,  50, 250, 250],
             [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
             [250, 250, 250, 250, 250, 255, 255, 255, 255, 255],
             [250, 250, 250, 250, 250, 255, 255, 255, 255, 255],
             [250, 120, 250, 250, 250, 255, 180, 180, 180, 255],
             [250, 250, 250, 250, 250, 255, 180, 180, 180, 255],
             [250, 250, 250, 250, 250, 255, 255, 255, 255, 255]], 
            dtype=np.uint8)
        data_float = data.astype(np.double) / 255.0
        output = criteria.area_closing(data_float, 4)

        output_8bit = 255.0 * output
        output_8bit = output_8bit.astype(np.uint8)

        expected = np.array(
            [[250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
             [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
             [250, 250, 100, 100, 250, 250, 250, 250, 250, 250],
             [250, 250, 100, 100, 250, 250, 250, 250, 250, 250],
             [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
             [250, 250, 250, 250, 250, 255, 255, 255, 255, 255],
             [250, 250, 250, 250, 250, 255, 255, 255, 255, 255],
             [250, 250, 250, 250, 250, 255, 180, 180, 180, 255],
             [250, 250, 250, 250, 250, 255, 180, 180, 180, 255],
             [250, 250, 250, 250, 250, 255, 255, 255, 255, 255]], 
            dtype=np.uint8)

        pdb.set_trace()

        error = diff(output_8bit, expected)
        assert error < eps
