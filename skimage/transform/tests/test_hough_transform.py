import numpy as np
from numpy.testing import *

import skimage.transform as tf
import skimage.transform.hough_transform as ht
from skimage.draw import circle_perimeter, line


def append_desc(func, description):
    """Append the test function ``func`` and append
    ``description`` to its name.
    """
    func.description = func.__module__ + '.' + func.__name__ + description

    return func


def test_hough_line():
    # Generate a test image
    img = np.zeros((100, 150), dtype=int)
    rr, cc = line(60, 130, 80, 10)
    img[rr, cc] = 1

    out, angles, d = tf.hough_line(img)

    y, x = np.where(out == out.max())
    dist = d[y[0]]
    theta = angles[x[0]]

    assert_almost_equal(dist, 80.723, 1)
    assert_almost_equal(theta, 1.41, 1)


def test_hough_line_angles():
    img = np.zeros((10, 10))
    img[0, 0] = 1

    out, angles, d = tf.hough_line(img, np.linspace(0, 360, 10))

    assert_equal(len(angles), 10)


def test_probabilistic_hough():
    # Generate a test image
    img = np.zeros((100, 100), dtype=int)
    for i in range(25, 75):
        img[100 - i, i] = 100
        img[i, i] = 100
    # decrease default theta sampling because similar orientations may confuse
    # as mentioned in article of Galambos et al
    theta = np.linspace(0, np.pi, 45)
    lines = tf.probabilistic_hough_line(img, threshold=10, line_length=10,
                                line_gap=1, theta=theta)
    # sort the lines according to the x-axis
    sorted_lines = []
    for line in lines:
        line = list(line)
        line.sort(key=lambda x: x[0])
        sorted_lines.append(line)
    assert([(25, 75), (74, 26)] in sorted_lines)
    assert([(25, 25), (74, 74)] in sorted_lines)


def test_hough_line_peaks():
    img = np.zeros((100, 150), dtype=int)
    rr, cc = line(60, 130, 80, 10)
    img[rr, cc] = 1

    out, angles, d = tf.hough_line(img)

    out, theta, dist = tf.hough_line_peaks(out, angles, d)

    assert_equal(len(dist), 1)
    assert_almost_equal(dist[0], 80.723, 1)
    assert_almost_equal(theta[0], 1.41, 1)


def test_hough_line_peaks_dist():
    img = np.zeros((100, 100), dtype=np.bool_)
    img[:, 30] = True
    img[:, 40] = True
    hspace, angles, dists = tf.hough_line(img)
    assert len(tf.hough_line_peaks(hspace, angles, dists, min_distance=5)[0]) == 2
    assert len(tf.hough_line_peaks(hspace, angles, dists, min_distance=15)[0]) == 1


def test_hough_line_peaks_angle():
    img = np.zeros((100, 100), dtype=np.bool_)
    img[:, 0] = True
    img[0, :] = True

    hspace, angles, dists = tf.hough_line(img)
    assert len(tf.hough_line_peaks(hspace, angles, dists, min_angle=45)[0]) == 2
    assert len(tf.hough_line_peaks(hspace, angles, dists, min_angle=90)[0]) == 1

    theta = np.linspace(0, np.pi, 100)
    hspace, angles, dists = tf.hough_line(img, theta)
    assert len(tf.hough_line_peaks(hspace, angles, dists, min_angle=45)[0]) == 2
    assert len(tf.hough_line_peaks(hspace, angles, dists, min_angle=90)[0]) == 1

    theta = np.linspace(np.pi / 3, 4. / 3 * np.pi, 100)
    hspace, angles, dists = tf.hough_line(img, theta)
    assert len(tf.hough_line_peaks(hspace, angles, dists, min_angle=45)[0]) == 2
    assert len(tf.hough_line_peaks(hspace, angles, dists, min_angle=90)[0]) == 1


def test_hough_line_peaks_num():
    img = np.zeros((100, 100), dtype=np.bool_)
    img[:, 30] = True
    img[:, 40] = True
    hspace, angles, dists = tf.hough_line(img)
    assert len(tf.hough_line_peaks(hspace, angles, dists, min_distance=0,
                              min_angle=0, num_peaks=1)[0]) == 1


def test_hough_circle():
    # Prepare picture
    img = np.zeros((120, 100), dtype=int)
    radius = 20
    x_0, y_0 = (99, 50)
    x, y = circle_perimeter(y_0, x_0, radius)
    img[y, x] = 1

    out = tf.hough_circle(img, np.array([radius]))

    x, y = np.where(out[0] == out[0].max())
    # Offset for x_0, y_0
    assert_equal(x[0], x_0 + radius)
    assert_equal(y[0], y_0 + radius)

if __name__ == "__main__":
    run_module_suite()
