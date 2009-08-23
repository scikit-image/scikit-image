import numpy as np
from numpy.testing import *

from scikits.image.transform import *

def test_hough():
    # Generate a test image
    img = np.zeros((100, 150), dtype=bool)
    img[30, :] = 1
    img[:, 65] = 1
    img[35:45, 35:50] = 1
    for i in range(90):
        img[i, i] = 1

    out, angles, d = hough(img)

    assert_equal(out.max(), 100)
    assert_equal(len(angles), 180)
