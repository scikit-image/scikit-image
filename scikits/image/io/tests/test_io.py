import os.path

import numpy as np
from numpy.testing import *

from scikits.image import data_dir
from scikits.image.io import io


class TestImageCollection():
    pattern = [os.path.join(data_dir, pic) for pic in ['camera.png',
                                                       'color.png']]

    def setUp(self):
        self.collection = io.ImageCollection(self.pattern)

    def test_len(self):
        assert len(self.collection) == 2

    def test_getitem(self):
        num = len(self.collection)
        for i in range(-num, num):
            assert type(self.collection[i]) is np.ndarray
        assert_array_almost_equal(self.collection[0],
                                  self.collection[-num])

        #assert_raises expects a callable, hence this do-very-little func
        def return_img(n):
            return self.collection[n]
        assert_raises(IndexError, return_img, num)
        assert_raises(IndexError, return_img, -num-1)

    def test_files_property(self):
        assert isinstance(self.collection.files, list)

        def set_files(f):
            self.collection.files = f
        assert_raises(AttributeError, set_files, 'newfiles')

    def test_as_grey_property(self):
        self.collection.as_grey = False
        assert self.collection[1].ndim == 3
        self.collection.as_grey = True
        assert self.collection[1].ndim == 2


if __name__ == "__main__":
    run_module_suite()

