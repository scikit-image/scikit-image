import numpy as np
from skimage.draw import circle
from skimage.feature import blob_dog, blob_log
import math


def test_blob_dog():
    r2 = math.sqrt(2)
    img = np.ones((512, 512))

    xs, ys = circle(400, 130, 5)
    img[xs, ys] = 255

    xs, ys = circle(100, 300, 25)
    img[xs, ys] = 255

    xs, ys = circle(200, 350, 45)
    img[xs, ys] = 255

    blobs = blob_dog(img, min_sigma=5, max_sigma=50)
    radius = lambda x: r2*x[2]
    s = sorted(blobs, key=radius)
    thresh = 5

    b = s[0]
    assert abs(b[0] - 400) <= thresh
    assert abs(b[1] - 130) <= thresh
    assert abs(radius(b) - 5) <= thresh

    b = s[1]
    print abs(radius(b))
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 300) <= thresh
    assert abs(radius(b) - 25) <= thresh

    b = s[2]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 350) <= thresh
    assert abs(radius(b) - 45) <= thresh


def test_blob_log():
    r2 = math.sqrt(2)
    img = np.ones((512, 512))

    xs, ys = circle(400, 130, 5)
    img[xs, ys] = 255

    xs, ys = circle(160, 50, 15)
    img[xs, ys] = 255

    xs, ys = circle(100, 300, 25)
    img[xs, ys] = 255

    xs, ys = circle(200, 350, 30)
    img[xs, ys] = 255

    blobs = blob_log(img, min_sigma=5, max_sigma=20, threshold=1)

    radius = lambda x: r2*x[2]
    s = sorted(blobs, key=radius)
    thresh = 3

    b = s[0]
    assert abs(b[0] - 400) <= thresh
    assert abs(b[1] - 130) <= thresh
    assert abs(radius(b) - 5) <= thresh

    b = s[1]
    assert abs(b[0] - 160) <= thresh
    assert abs(b[1] - 50) <= thresh
    assert abs(radius(b) - 15) <= thresh

    b = s[2]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 300) <= thresh
    assert abs(radius(b) - 25) <= thresh

    b = s[3]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 350) <= thresh
    assert abs(radius(b) - 30) <= thresh
