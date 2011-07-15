import numpy as np
from numpy.testing import *

import scikits.image.transform as tf
import scikits.image.transform.hough_transform as ht

def append_desc(func, description):
    """Append the test function ``func`` and append
    ``description`` to its name.
    """
    func.description = func.__module__ + '.' + func.func_name + description

    return func

def test_hough():
    # Generate a test image
    img = np.zeros((100, 100), dtype=int)
    for i in range(25, 75):
        img[100 - i, i] = 1

    out, angles, d = tf.hough(img)

    y, x = np.where(out == out.max())
    dist = d[y[0]]
    theta = angles[x[0]]

    assert_equal(dist > 70, dist < 72)
    assert_equal(theta > 0.78, theta < 0.79)

def test_hough_angles():
    img = np.zeros((10, 10))
    img[0, 0] = 1

    out, angles, d = tf.hough(img, np.linspace(0, 360, 10))

    assert_equal(len(angles), 10)

def test_py_hough():
    ht._hough, fast_hough = ht._py_hough, ht._hough

    yield append_desc(test_hough, '_python')
    yield append_desc(test_hough_angles, '_python')

    tf._hough = fast_hough

if __name__ == "__main__":
    run_module_suite()

