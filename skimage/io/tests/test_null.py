import os
import warnings
from contextlib import contextmanager

import numpy as np
from numpy.testing import raises

from skimage import io
from skimage import data_dir


@contextmanager
def warnings_as_errors():
    # Temporarily set warnings as errors so we can test the warning is raised.
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        yield

@raises(Warning)
def test_null_imread():
    path = os.path.join(data_dir, 'color.png')
    with warnings_as_errors():
        io.imread(path, plugin='null')


@raises(Warning)
def test_null_imsave():
    with warnings_as_errors():
        io.imsave('dummy.png', np.zeros((3, 3)), plugin='null')


@raises(Warning)
def test_null_imshow():
    with warnings_as_errors():
        io.imshow(np.zeros((3, 3)), plugin='null')


@raises(Warning)
def test_null_imread_collection():
    # Note that the null plugin doesn't define an `imread_collection` plugin
    # but this function is dynamically added by the plugin manager.
    path = os.path.join(data_dir, '*.png')
    with warnings_as_errors():
        collection = io.imread_collection(path, plugin='null')
        collection[0]


if __name__ == '__main__':
    from numpy.testing import run_module_suite
    run_module_suite()
