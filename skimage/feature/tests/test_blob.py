import numpy as np
from skimage.draw import circle
from skimage.feature import blob_dog, blob_log, blob_doh
import math
from numpy.testing import assert_raises


def test_blob_dog():
    r2 = math.sqrt(2)
    img = np.ones((512, 512))
    img3 = np.ones((5, 5, 5))

    xs, ys = circle(400, 130, 5)
    img[xs, ys] = 255

    xs, ys = circle(100, 300, 25)
    img[xs, ys] = 255

    xs, ys = circle(200, 350, 45)
    img[xs, ys] = 255

    blobs = blob_dog(img, min_sigma=5, max_sigma=50)
    radius = lambda x: r2 * x[2]
    s = sorted(blobs, key=radius)
    thresh = 5

    b = s[0]
    assert abs(b[0] - 400) <= thresh
    assert abs(b[1] - 130) <= thresh
    assert abs(radius(b) - 5) <= thresh

    b = s[1]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 300) <= thresh
    assert abs(radius(b) - 25) <= thresh

    b = s[2]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 350) <= thresh
    assert abs(radius(b) - 45) <= thresh

    assert_raises(ValueError, blob_dog, img3)

    # Testing no peaks
    img_empty = np.zeros((100,100))
    assert blob_dog(img_empty).size == 0


def test_blob_log():
    r2 = math.sqrt(2)
    img = np.ones((256, 256))
    img3 = np.ones((5, 5, 5))

    xs, ys = circle(200, 65, 5)
    img[xs, ys] = 255

    xs, ys = circle(80, 25, 15)
    img[xs, ys] = 255

    xs, ys = circle(50, 150, 25)
    img[xs, ys] = 255

    xs, ys = circle(100, 175, 30)
    img[xs, ys] = 255

    blobs = blob_log(img, min_sigma=5, max_sigma=20, threshold=1)

    radius = lambda x: r2 * x[2]
    s = sorted(blobs, key=radius)
    thresh = 3

    b = s[0]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 65) <= thresh
    assert abs(radius(b) - 5) <= thresh

    b = s[1]
    assert abs(b[0] - 80) <= thresh
    assert abs(b[1] - 25) <= thresh
    assert abs(radius(b) - 15) <= thresh

    b = s[2]
    assert abs(b[0] - 50) <= thresh
    assert abs(b[1] - 150) <= thresh
    assert abs(radius(b) - 25) <= thresh

    b = s[3]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 175) <= thresh
    assert abs(radius(b) - 30) <= thresh

    # Testing log scale
    blobs = blob_log(
        img,
        min_sigma=5,
        max_sigma=20,
        threshold=1,
        log_scale=True)

    b = s[0]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 65) <= thresh
    assert abs(radius(b) - 5) <= thresh

    b = s[1]
    assert abs(b[0] - 80) <= thresh
    assert abs(b[1] - 25) <= thresh
    assert abs(radius(b) - 15) <= thresh

    b = s[2]
    assert abs(b[0] - 50) <= thresh
    assert abs(b[1] - 150) <= thresh
    assert abs(radius(b) - 25) <= thresh

    b = s[3]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 175) <= thresh
    assert abs(radius(b) - 30) <= thresh

    assert_raises(ValueError, blob_log, img3)

    # Testing no peaks
    img_empty = np.zeros((100,100))
    assert blob_log(img_empty).size == 0


def test_blob_doh():
    img = np.ones((512, 512), dtype=np.uint8)
    img3 = np.ones((5, 5, 5))

    xs, ys = circle(400, 130, 20)
    img[xs, ys] = 255

    xs, ys = circle(460, 50, 30)
    img[xs, ys] = 255

    xs, ys = circle(100, 300, 40)
    img[xs, ys] = 255

    xs, ys = circle(200, 350, 50)
    img[xs, ys] = 255

    blobs = blob_doh(
        img,
        min_sigma=1,
        max_sigma=60,
        num_sigma=10,
        threshold=.05)

    radius = lambda x: x[2]
    s = sorted(blobs, key=radius)
    thresh = 4

    b = s[0]
    assert abs(b[0] - 400) <= thresh
    assert abs(b[1] - 130) <= thresh
    assert abs(radius(b) - 20) <= thresh

    b = s[1]
    assert abs(b[0] - 460) <= thresh
    assert abs(b[1] - 50) <= thresh
    assert abs(radius(b) - 30) <= thresh

    b = s[2]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 300) <= thresh
    assert abs(radius(b) - 40) <= thresh

    b = s[3]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 350) <= thresh
    assert abs(radius(b) - 50) <= thresh

    # Testing log scale
    blobs = blob_doh(
        img,
        min_sigma=1,
        max_sigma=60,
        num_sigma=10,
        log_scale=True,
        threshold=.05)

    b = s[0]
    assert abs(b[0] - 400) <= thresh
    assert abs(b[1] - 130) <= thresh
    assert abs(radius(b) - 20) <= thresh

    b = s[1]
    assert abs(b[0] - 460) <= thresh
    assert abs(b[1] - 50) <= thresh
    assert abs(radius(b) - 30) <= thresh

    b = s[2]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 300) <= thresh
    assert abs(radius(b) - 40) <= thresh

    b = s[3]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 350) <= thresh
    assert abs(radius(b) - 50) <= thresh

    assert_raises(ValueError, blob_doh, img3)

    # Testing no peaks
    img_empty = np.zeros((100,100))
    assert blob_doh(img_empty).size == 0


def test_blob_overlap():
    img = np.ones((512, 512), dtype=np.uint8)

    xs, ys = circle(100, 100, 20)
    img[xs, ys] = 255

    xs, ys = circle(120, 100, 30)
    img[xs, ys] = 255

    blobs = blob_doh(
        img,
        min_sigma=1,
        max_sigma=60,
        num_sigma=10,
        threshold=.05)

    assert len(blobs) == 1
