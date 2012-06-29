import numpy as np

from skimage import data
from skimage import img_as_float

from skimage.feature import harris


def test_square_image():
    im = np.zeros((50, 50)).astype(float)
    im[:25, :25] = 1.
    results = harris(im)
    assert results.any()
    assert len(results) == 1


def test_noisy_square_image():
    im = np.zeros((50, 50)).astype(float)
    im[:25, :25] = 1.
    im = im + np.random.uniform(size=im.shape) * .5
    results = harris(im)
    assert results.any()
    assert len(results) == 1


def test_squared_dot():
    im = np.zeros((50, 50))
    im[4:8, 4:8] = 1
    im = img_as_float(im)
    results = harris(im, min_distance=3)
    assert (results == np.array([[6, 6]])).all()


def test_rotated_lena():
    """
    The harris filter should yield the same results with an image and it's
    rotation.
    """
    im = img_as_float(data.lena().mean(axis=2))
    results = harris(im)
    im_rotated = im.T
    results_rotated = harris(im_rotated)
    assert (np.sort(results[:, 0]) == np.sort(results_rotated[:, 1])).all()
    assert (np.sort(results[:, 1]) == np.sort(results_rotated[:, 0])).all()


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
