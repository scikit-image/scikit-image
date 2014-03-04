import numpy as np
from skimage.draw import circle
from skimage.feature import blob_dog


def test_blob_dog():
    img = np.ones((512, 512))

    xs, ys = circle(400, 130, 5)
    img[xs, ys] = 255

    xs, ys = circle(100, 300, 25)
    img[xs, ys] = 255

    xs, ys = circle(200, 350, 30)
    img[xs, ys] = 255

    blobs = blob_dog(img)
    coords = blobs[:, 0:2]

    if coords[coords == [400, 130]].shape != (2,):
        assert False

    if coords[coords == [100, 300]].shape != (2,):
        assert False

    if coords[coords == [200, 350]].shape != (2,):
        assert False
