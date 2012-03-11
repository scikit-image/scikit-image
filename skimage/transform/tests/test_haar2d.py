import numpy as np
from numpy.testing import *

import skimage.data as data
from skimage.transform import haar2d, ihaar2d

image = data.camera()

def test_image():
    assert_equal(image.shape, (512,512))

def test_basic():
    h = haar2d(image)
    ih = ihaar2d(h)
    assert_equal(image, ih)

def test_advanced():
    h = haar2d(image[:510,:510], 4)
    ih = ihaar2d(h, 4)
    assert_equal(image[:510,:510], ih[:510,:510])

if __name__ == '__main__':
    run_module_suite()
