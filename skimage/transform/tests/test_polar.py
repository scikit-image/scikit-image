import numpy as np
from skimage.draw import circle
from skimage.transform import cart2pol
from numpy.testing import assert_equal


def test_cart2pol_centered_disk():
    img = np.zeros((500, 500), dtype=np.uint8)
    rr, cc = circle(250, 250, 100)
    img[rr, cc] = 1
    polimg = cart2pol(img)

    assert_equal(polimg[:99, :], np.ones(polimg[:99, :].shape))
    assert_equal(polimg[101:, :], np.zeros(polimg[101:, :].shape))
    assert_equal(polimg.shape, (250, 360))


def test_cart2pol_offcentered_disk():
    img = np.zeros((500, 500), dtype=np.uint8)
    rr, cc = circle(250, 150, 100)
    img[rr, cc] = 1
    polimg = cart2pol(img, center=(150, 250))

    assert_equal(polimg[:99, :], np.ones(polimg[:99, :].shape))
    assert_equal(polimg[101:, :], np.zeros(polimg[101:, :].shape))
    assert_equal(polimg.shape, (150, 360))


def test_cart2pol_offcentered_disk_fulloutput():
    img = np.zeros((500, 500), dtype=np.uint8)
    rr, cc = circle(250, 150, 100)
    img[rr, cc] = 1
    polimg = cart2pol(img, center=(150, 250), full_output=True)

    assert_equal(polimg[:99, :], np.ones(polimg[:99, :].shape))
    assert_equal(polimg[101:, :], np.zeros(polimg[101:, :].shape))


def test_cart2pol_rgb():
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    rr, cc = circle(250, 150, 100)
    img[rr, cc] = (51, 101, 201)
    polimg = cart2pol(img, center=(150, 250))

    assert_equal(polimg[:99, :, 0], np.ones(polimg[:99, :, 0].shape) * 51)
    assert_equal(polimg[:99, :, 1], np.ones(polimg[:99, :, 1].shape) * 101)
    assert_equal(polimg[:99, :, 2], np.ones(polimg[:99, :, 2].shape) * 201)
    assert_equal(polimg[101:, :, :], np.zeros(polimg[101:, :, :].shape))


def test_cart2pol_rgb_fulloutput():
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    rr, cc = circle(250, 150, 100)
    img[rr, cc] = (51, 101, 201)
    polimg = cart2pol(img, center=(150, 250), full_output=True)

    assert_equal(polimg[:99, :, 0], np.ones(polimg[:99, :, 0].shape) * 51)
    assert_equal(polimg[:99, :, 1], np.ones(polimg[:99, :, 1].shape) * 101)
    assert_equal(polimg[:99, :, 2], np.ones(polimg[:99, :, 2].shape) * 201)
    assert_equal(polimg[101:, :, :], np.zeros(polimg[101:, :, :].shape))
