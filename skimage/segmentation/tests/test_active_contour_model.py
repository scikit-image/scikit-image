import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour

from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_allclose
from skimage._shared._warnings import expected_warnings


def test_periodic_reference():
    img = data.astronaut()
    img = rgb2gray(img)
    s = np.linspace(0, 2*np.pi, 400)
    r = 100 + 100*np.sin(s)
    c = 220 + 100*np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3), init, alpha=0.015, beta=10,
                           w_line=0, w_edge=1, gamma=0.001, coordinates='rc')
    refr = [98, 99, 100, 101, 102, 103, 104, 105, 106, 108]
    refc = [299, 298, 298, 298, 298, 297, 297, 296, 296, 295]
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)


def test_fixed_reference():
    img = data.text()
    r = np.linspace(136, 50, 100)
    c = np.linspace(5, 424, 100)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 1), init, boundary_condition='fixed',
                           alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1,
                           coordinates='rc')
    refr = [136, 135, 134, 133, 132, 131, 129, 128, 127, 125]
    refc = [5, 9, 13, 17, 21, 25, 30, 34, 38, 42]
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)


def test_free_reference():
    img = data.text()
    r = np.linspace(70, 40, 100)
    c = np.linspace(5, 424, 100)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3), init, boundary_condition='free',
                           alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1,
                           coordinates='rc')
    refr = [76, 76, 75, 74, 73, 72, 71, 70, 69, 69]
    refc = [10, 13, 16, 19, 23, 26, 29, 32, 36, 39]
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)


def test_RGB():
    img = gaussian(data.text(), 1)
    imgR = np.zeros((img.shape[0], img.shape[1], 3))
    imgG = np.zeros((img.shape[0], img.shape[1], 3))
    imgRGB = np.zeros((img.shape[0], img.shape[1], 3))
    imgR[:, :, 0] = img
    imgG[:, :, 1] = img
    imgRGB[:, :, :] = img[:, :, None]
    r = np.linspace(136, 50, 100)
    c = np.linspace(5, 424, 100)
    init = np.array([r, c]).T
    snake = active_contour(imgR, init, boundary_condition='fixed',
                           alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1,
                           coordinates='rc')
    refr = [136, 135, 134, 133, 132, 131, 129, 128, 127, 125]
    refc = [5, 9, 13, 17, 21, 25, 30, 34, 38, 42]
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)
    snake = active_contour(imgG, init, boundary_condition='fixed',
                           alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1,
                           coordinates='rc')
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)
    snake = active_contour(imgRGB, init, boundary_condition='fixed',
                           alpha=0.1, beta=1.0, w_line=-5/3., w_edge=0,
                           gamma=0.1, coordinates='rc')
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)


def test_end_points():
    img = data.astronaut()
    img = rgb2gray(img)
    s = np.linspace(0, 2*np.pi, 400)
    r = 100 + 100*np.sin(s)
    c = 220 + 100*np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3), init,
                           boundary_condition='periodic', alpha=0.015, beta=10,
                           w_line=0, w_edge=1, gamma=0.001, max_iterations=100,
                           coordinates='rc')
    assert np.sum(np.abs(snake[0, :]-snake[-1, :])) < 2
    snake = active_contour(gaussian(img, 3), init,
                           boundary_condition='free', alpha=0.015, beta=10,
                           w_line=0, w_edge=1, gamma=0.001, max_iterations=100,
                           coordinates='rc')
    assert np.sum(np.abs(snake[0, :]-snake[-1, :])) > 2
    snake = active_contour(gaussian(img, 3), init,
                           boundary_condition='fixed', alpha=0.015, beta=10,
                           w_line=0, w_edge=1, gamma=0.001, max_iterations=100,
                           coordinates='rc')
    assert_allclose(snake[0, :], [r[0], c[0]], atol=1e-5)


def test_bad_input():
    img = np.zeros((10, 10))
    r = np.linspace(136, 50, 100)
    c = np.linspace(5, 424, 100)
    init = np.array([r, c]).T
    with testing.raises(ValueError):
        active_contour(img, init, boundary_condition='wrong',
                       coordinates='rc')
    with testing.raises(ValueError):
        active_contour(img, init, max_iterations=-15,
                       coordinates='rc')


def test_bc_deprecation():
    with expected_warnings(['boundary_condition']):
        img = rgb2gray(data.astronaut())
        s = np.linspace(0, 2*np.pi, 400)
        r = 100 + 100*np.sin(s)
        c = 220 + 100*np.cos(s)
        init = np.array([r, c]).T
        snake = active_contour(gaussian(img, 3), init,
                               bc='periodic', alpha=0.015, beta=10,
                               w_line=0, w_edge=1, gamma=0.001,
                               max_iterations=100, coordinates='rc')


def test_xy_coord_warning():
    # this should raise ValueError after 0.18.
    with expected_warnings(['xy coordinates']):
        img = rgb2gray(data.astronaut())
        s = np.linspace(0, 2*np.pi, 400)
        x = 100 + 100*np.sin(s)
        y = 220 + 100*np.cos(s)
        init = np.array([x, y]).T
        snake = active_contour(gaussian(img, 3), init,
                               boundary_condition='periodic', alpha=0.015,
                               beta=10, w_line=0, w_edge=1, gamma=0.001,
                               max_iterations=100)
