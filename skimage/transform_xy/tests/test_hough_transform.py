import numpy as np

from skimage.transform_xy import hough_line, probabilistic_hough_line

from skimage import data, draw

from skimage._shared import testing
from skimage._shared.testing import (assert_almost_equal, assert_equal,
                                     test_parallel)


@test_parallel()
def test_hough_line():
    # Generate a test image
    img = np.zeros((100, 150), dtype=int)
    rr, cc = draw.line(60, 130, 80, 10)
    img[rr, cc] = 1

    out, angles, d = hough_line(img)

    y, x = np.where(out == out.max())
    dist = d[y[0]]
    theta = angles[x[0]]

    assert_almost_equal(dist, 80.723, 1)
    assert_almost_equal(theta, 1.41, 1)


def test_hough_line_angles():
    img = np.zeros((10, 10))
    img[0, 0] = 1

    out, angles, d = hough_line(img, np.linspace(0, 360, 10))

    assert_equal(len(angles), 10)


def test_hough_line_bad_input():
    img = np.zeros(100)
    img[10] = 1

    # Expected error, img must be 2D
    with testing.raises(ValueError):
        hough_line(img, np.linspace(0, 360, 10))


def test_probabilistic_hough():
    # Generate a test image
    img = np.zeros((100, 100), dtype=int)
    for i in range(25, 75):
        img[100 - i, i] = 100
        img[i, i] = 100

    # decrease default theta sampling because similar orientations may confuse
    # as mentioned in article of Galambos et al
    theta = np.linspace(0, np.pi, 45)
    lines = probabilistic_hough_line(
        img, threshold=10, line_length=10, line_gap=1, theta=theta)
    # sort the lines according to the x-axis
    sorted_lines = []
    for line in lines:
        line = list(line)
        line.sort(key=lambda x: x[0])
        sorted_lines.append(line)

    assert([(25, 75), (74, 26)] in sorted_lines)
    assert([(25, 25), (74, 74)] in sorted_lines)

    # Execute with default theta
    probabilistic_hough_line(img, line_length=10, line_gap=3)


def test_probabilistic_hough_seed():
    # Load image that is likely to give a randomly varying number of lines
    image = data.checkerboard()

    # Use constant seed to ensure a deterministic output
    lines = probabilistic_hough_line(image, threshold=50,
                                     line_length=50, line_gap=1,
                                     seed=1234)
    assert len(lines) == 65


def test_probabilistic_hough_bad_input():
    img = np.zeros(100)
    img[10] = 1

    # Expected error, img must be 2D
    with testing.raises(ValueError):
        probabilistic_hough_line(img)


