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



def test_area_closing():
    "Adding/subtracting a constant and clipping"
    data = np.ones((10, 10), dtype=np.uint8) * 250
    data[5:,5:] += 5
    data[2:4,2:4] = 100
    data[2:4, 7:8] = 50
    data[7, 1] = 120
    data[7:9, 6:9] = 180
    data_float = data.astype(np.double) / 255.0
    pdb.set_trace()
    output = criteria.area_closing(data_float, 4)
    
    error = diff(img_constant_subtracted, expected)
    assert error < eps

test_area_closing()

