import numpy as np

from skimage import transform


image_shape = (512, 512)
polygon = np.array([[80, 111, 146, 234, 407, 300, 187, 45],
                    [465, 438, 499, 380, 450, 287, 210, 167]]).T


def test_poly2mask_default():
    mask = transform.poly2mask(image_shape, polygon, backend='default')
    assert mask.shape == image_shape
    assert mask.sum() == 57647


def test_poly2mask_matplotlib():
    mask = transform.poly2mask(image_shape, polygon, backend='matplotlib')
    assert mask.shape == image_shape
    assert mask.sum() == 57650
