import os

from numpy.testing import *
import numpy as np
from StringIO import StringIO

import skimage.io as io
from skimage import data_dir


def test_stack_basic():
    x = np.arange(12).reshape(3, 4)
    io.push(x)

    assert_array_equal(io.pop(), x)


@raises(ValueError)
def test_stack_non_array():
    io.push([[1, 2, 3]])


def test_imread_url():
    # tweak data path so that file URI works on both unix and windows.
    data_path = data_dir.lstrip(os.path.sep)
    data_path = data_path.replace(os.path.sep, '/')
    image_url = 'file:///{0}/camera.png'.format(data_path)
    image = io.imread(image_url)
    assert image.shape == (512, 512)

def test_imsave_filelike():
    image = np.array([[0, 0], [0, 0]], dtype=float)
    s = StringIO()

    # save to file-like object
    io.imsave(s, image)

    # read from file-like object
    s.seek(0)
    image = io.imread(s)
    assert image.shape == (2, 2)

if __name__ == "__main__":
    run_module_suite()
