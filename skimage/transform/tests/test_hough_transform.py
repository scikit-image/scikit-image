import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

import skimage.transform as tf
from skimage.draw import line, circle_perimeter, ellipse_perimeter
from skimage._shared._warnings import expected_warnings


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

    with expected_warnings(['`background`']):
        out, theta, dist = tf.hough_line_peaks(out, angles, d)

    assert_equal(len(dist), 1)
    assert_almost_equal(dist[0], 80.723, 1)
    assert_almost_equal(theta[0], 1.41, 1)


def test_hough_line_peaks_dist():
    img = np.zeros((100, 100), dtype=np.bool_)
    img[:, 30] = True
    img[:, 40] = True
    hspace, angles, dists = tf.hough_line(img)
    with expected_warnings(['`background`']):
        assert len(tf.hough_line_peaks(hspace, angles, dists,
                                       min_distance=5)[0]) == 2
        assert len(tf.hough_line_peaks(hspace, angles, dists,
                                       min_distance=15)[0]) == 1


def test_hough_line_peaks_angle():
    with expected_warnings(['`background`']):
        check_hough_line_peaks_angle()


def check_hough_line_peaks_angle():
    img = np.zeros((100, 100), dtype=np.bool_)
    img[:, 0] = True
    img[0, :] = True

    hspace, angles, dists = tf.hough_line(img)
    assert len(tf.hough_line_peaks(hspace, angles, dists,
                                   min_angle=45)[0]) == 2
    assert len(tf.hough_line_peaks(hspace, angles, dists,
                                   min_angle=90)[0]) == 1

    theta = np.linspace(0, np.pi, 100)
    hspace, angles, dists = tf.hough_line(img, theta)
    assert len(tf.hough_line_peaks(hspace, angles, dists,
                                   min_angle=45)[0]) == 2
    assert len(tf.hough_line_peaks(hspace, angles, dists,
                                   min_angle=90)[0]) == 1

    theta = np.linspace(np.pi / 3, 4. / 3 * np.pi, 100)
    hspace, angles, dists = tf.hough_line(img, theta)
    assert len(tf.hough_line_peaks(hspace, angles, dists,
                                   min_angle=45)[0]) == 2
    assert len(tf.hough_line_peaks(hspace, angles, dists,
                                   min_angle=90)[0]) == 1


def test_hough_line_peaks_num():
    img = np.zeros((100, 100), dtype=np.bool_)
    img[:, 30] = True
    img[:, 40] = True
    hspace, angles, dists = tf.hough_line(img)
    with expected_warnings(['`background`']):
        assert len(tf.hough_line_peaks(hspace, angles, dists, min_distance=0,
                                       min_angle=0, num_peaks=1)[0]) == 1


def test_hough_circle():
    # Prepare picture
    img = np.zeros((120, 100), dtype=int)
    radius = 20
    x_0, y_0 = (99, 50)
    y, x = circle_perimeter(y_0, x_0, radius)
    img[x, y] = 1

    out = tf.hough_circle(img, np.array([radius], dtype=np.intp))

    x, y = np.where(out[0] == out[0].max())
    assert_equal(x[0], x_0)
    assert_equal(y[0], y_0)


def test_hough_circle_extended():
    # Prepare picture
    # The circle center is outside the image
    img = np.zeros((100, 100), dtype=int)
    radius = 20
    x_0, y_0 = (-5, 50)
    y, x = circle_perimeter(y_0, x_0, radius)
    img[x[np.where(x > 0)], y[np.where(x > 0)]] = 1

    out = tf.hough_circle(img, np.array([radius], dtype=np.intp),
                          full_output=True)

    x, y = np.where(out[0] == out[0].max())
    # Offset for x_0, y_0
    assert_equal(x[0], x_0 + radius)
    assert_equal(y[0], y_0 + radius)


def test_hough_ellipse_zero_angle():
    img = np.zeros((25, 25), dtype=int)
    rx = 6
    ry = 8
    x0 = 12
    y0 = 15
    angle = 0
    rr, cc = ellipse_perimeter(y0, x0, ry, rx)
    img[rr, cc] = 1
    result = tf.hough_ellipse(img, threshold=9)
    best = result[-1]
    assert_equal(best[1], y0)
    assert_equal(best[2], x0)
    assert_almost_equal(best[3], ry, decimal=1)
    assert_almost_equal(best[4], rx, decimal=1)
    assert_equal(best[5], angle)
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_posangle1():
    # ry > rx, angle in [0:pi/2]
    img = np.zeros((30, 24), dtype=int)
    rx = 6
    ry = 12
    x0 = 10
    y0 = 15
    angle = np.pi / 1.35
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = tf.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    assert_almost_equal(best[1] / 100., y0 / 100., decimal=1)
    assert_almost_equal(best[2] / 100., x0 / 100., decimal=1)
    assert_almost_equal(best[3] / 10., ry / 10., decimal=1)
    assert_almost_equal(best[4] / 100., rx / 100., decimal=1)
    assert_almost_equal(best[5], angle, decimal=1)
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_posangle2():
    # ry < rx, angle in [0:pi/2]
    img = np.zeros((30, 24), dtype=int)
    rx = 12
    ry = 6
    x0 = 10
    y0 = 15
    angle = np.pi / 1.35
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = tf.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    assert_almost_equal(best[1] / 100., y0 / 100., decimal=1)
    assert_almost_equal(best[2] / 100., x0 / 100., decimal=1)
    assert_almost_equal(best[3] / 10., ry / 10., decimal=1)
    assert_almost_equal(best[4] / 100., rx / 100., decimal=1)
    assert_almost_equal(best[5], angle, decimal=1)
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_posangle3():
    # ry < rx, angle in [pi/2:pi]
    img = np.zeros((30, 24), dtype=int)
    rx = 12
    ry = 6
    x0 = 10
    y0 = 15
    angle = np.pi / 1.35 + np.pi / 2.
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = tf.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_posangle4():
    # ry < rx, angle in [pi:3pi/4]
    img = np.zeros((30, 24), dtype=int)
    rx = 12
    ry = 6
    x0 = 10
    y0 = 15
    angle = np.pi / 1.35 + np.pi
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = tf.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_negangle1():
    # ry > rx, angle in [0:-pi/2]
    img = np.zeros((30, 24), dtype=int)
    rx = 6
    ry = 12
    x0 = 10
    y0 = 15
    angle = - np.pi / 1.35
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = tf.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_negangle2():
    # ry < rx, angle in [0:-pi/2]
    img = np.zeros((30, 24), dtype=int)
    rx = 12
    ry = 6
    x0 = 10
    y0 = 15
    angle = - np.pi / 1.35
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = tf.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_negangle3():
    # ry < rx, angle in [-pi/2:-pi]
    img = np.zeros((30, 24), dtype=int)
    rx = 12
    ry = 6
    x0 = 10
    y0 = 15
    angle = - np.pi / 1.35 - np.pi / 2.
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = tf.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_negangle4():
    # ry < rx, angle in [-pi:-3pi/4]
    img = np.zeros((30, 24), dtype=int)
    rx = 12
    ry = 6
    x0 = 10
    y0 = 15
    angle = - np.pi / 1.35 - np.pi
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = tf.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


if __name__ == "__main__":
    np.testing.run_module_suite()
