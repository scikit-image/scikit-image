import numpy as np
from skimage.draw import circle
from skimage.feature import get_blobs_dog
import math


def test_get_blobs_dog():
    img = np.ones((512, 512))

    xs, ys = circle(400, 130, 5)
    img[xs, ys] = 255

    xs, ys = circle(100, 300, 25)
    img[xs, ys] = 255

    xs, ys = circle(200, 350, 30)
    img[xs, ys] = 255

    # io.imshow(img)
    # plt.show()

    blobs = get_blobs_dog(img)
    area = lambda x: x[2]
    radius = lambda x: math.sqrt(x / math.pi)
    s = sorted(blobs, key=area)
    thresh = 5

    for b in s:
        print b[0], b[1], radius(b[2])

    b = s[0]
    assert abs(b[0] - 400) <= thresh
    assert abs(b[1] - 130) <= thresh
    assert abs(radius(b[2]) - 5) <= thresh

    b = s[1]
    print radius(b[2])
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 300) <= thresh
    assert abs(radius(b[2]) - 25) <= thresh

    b = s[2]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 350) <= thresh
    assert abs(radius(b[2]) - 30) <= thresh
