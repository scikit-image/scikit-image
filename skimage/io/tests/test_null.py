import os
import warnings

import numpy as np
from numpy.testing import raises

from skimage import io
from skimage import data_dir


@raises(Warning)
def test_null_imread():
    path = os.path.join(data_dir, 'color.png')
    with warnings.catch_warnings():  # Temporarily set warnings as errors.
        warnings.filterwarnings('error')
        io.imread(path, plugin='null')


@raises(Warning)
def test_null_imsave():
    with warnings.catch_warnings():  # Temporarily set warnings as errors.
        warnings.filterwarnings('error')
        io.imsave('dummy.png', np.zeros((3, 3)), plugin='null')


@raises(Warning)
def test_null_imshow():
    with warnings.catch_warnings():  # Temporarily set warnings as errors.
        warnings.filterwarnings('error')
        io.imshow(np.zeros((3, 3)), plugin='null')


if __name__ == '__main__':
    from numpy.testing import run_module_suite
    run_module_suite()
