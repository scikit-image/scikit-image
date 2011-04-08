import numpy as np
from numpy.testing import *

from scikits.image.transform import *

def test_hough():
    # Generate a test image
    img = np.zeros((100, 100), dtype=int)
    for i in range(25, 75):
        img[100 - i, i] = 1

    out, angles, d = hough(img)

    y, x = np.where(out == out.max())
    dist = d[y[0]]
    theta = angles[x[0]]

    assert_equal(dist > 70, dist < 72)
    assert_equal(theta > 0.78, theta < 0.79)

def test_hough_angles():
    img = np.zeros((10, 10))
    img[0, 0] = 1

    out, angles, d = hough(img, np.linspace(0, 360, 10))

    assert_equal(len(angles), 10)

if __name__ == "__main__":
    run_module_suite()

