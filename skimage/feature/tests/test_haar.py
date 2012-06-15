import numpy as np

from skimage import data
from skimage import img_as_float

from skimage.feature import haar


def test_square_image():
    im = np.zeros((50, 50)).astype(float)
    im[:25, :25] = 1.
    results = haar(im)
    assert len(results) == 1
