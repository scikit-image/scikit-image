import sys
import os.path

import numpy as np
from numpy.testing import *
from numpy.testing.decorators import skipif

from skimage import data_dir
from skimage.io import ImageCollection, MultiImage


try:
    from PIL import Image
except ImportError:
    PIL_available = False
else:
    PIL_available = True

if sys.version_info[0] > 2:
    basestring = str


class TestImageCollection():
    pattern = [os.path.join(data_dir, pic) for pic in ['camera.png',
                                                       'color.png']]

    def setUp(self):
        self.collection = ImageCollection(self.pattern)

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
        assert_raises(IndexError, return_img, -num - 1)

    def test_files_property(self):
        assert isinstance(self.collection.files, list)

        def set_files(f):
            self.collection.files = f
        assert_raises(AttributeError, set_files, 'newfiles')

    def test_custom_load(self):
        load_pattern = [(1, 'one'), (2, 'two')]

        def load_fn(x):
            return x

        ic = ImageCollection(load_pattern, load_func=load_fn)
        assert_equal(ic[1], (2, 'two'))


class TestMultiImage():

    def setUp(self):
        # This multipage TIF file was created with imagemagick:
        # convert im1.tif im2.tif -adjoin multipage.tif
        if PIL_available:
            self.img = MultiImage(os.path.join(data_dir, 'multipage.tif'))

    @skipif(not PIL_available)
    def test_len(self):
        assert len(self.img) == 2

    @skipif(not PIL_available)
    def test_getitem(self):
        num = len(self.img)
        for i in range(-num, num):
            assert type(self.img[i]) is np.ndarray
        assert_array_almost_equal(self.img[0],
                                  self.img[-num])

        #assert_raises expects a callable, hence this do-very-little func
        def return_img(n):
            return self.img[n]
        assert_raises(IndexError, return_img, num)
        assert_raises(IndexError, return_img, -num - 1)

    @skipif(not PIL_available)
    def test_files_property(self):
        assert isinstance(self.img.filename, basestring)

        def set_filename(f):
            self.img.filename = f
        assert_raises(AttributeError, set_filename, 'newfile')

    @skipif(not PIL_available)
    def test_conserve_memory_property(self):
        assert isinstance(self.img.conserve_memory, bool)

        def set_mem(val):
            self.img.conserve_memory = val
        assert_raises(AttributeError, set_mem, True)


if __name__ == "__main__":
    run_module_suite()
