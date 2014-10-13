import os

import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_allclose

from skimage import data_dir
from skimage.io.collection import MultiImage

import six


class TestMultiImage():

    def setUp(self):
        # This multipage TIF file was created with imagemagick:
        # convert im1.tif im2.tif -adjoin multipage.tif
        self.img = MultiImage(os.path.join(data_dir, 'multipage.tif'))

    def test_len(self):
        assert len(self.img) == 2

    def test_getitem(self):
        num = len(self.img)
        for i in range(-num, num):
            assert type(self.img[i]) is np.ndarray
        assert_allclose(self.img[0], self.img[-num])

        # assert_raises expects a callable, hence this thin wrapper function.
        def return_img(n):
            return self.img[n]
        assert_raises(IndexError, return_img, num)
        assert_raises(IndexError, return_img, -num - 1)

    def test_files_property(self):
        assert isinstance(self.img.filename, six.string_types)

        def set_filename(f):
            self.img.filename = f
        assert_raises(AttributeError, set_filename, 'newfile')

    def test_conserve_memory_property(self):
        assert isinstance(self.img.conserve_memory, bool)

        def set_mem(val):
            self.img.conserve_memory = val
        assert_raises(AttributeError, set_mem, True)

    def test_concatenate(self):
        array = self.img.concatenate()
        assert_equal(array.shape, (len(self.img),) + self.img[0].shape)


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
