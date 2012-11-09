import sys
print sys.path
import skimage
print skimage

import unittest

import numpy as np
from skimage.filter import rank

from skimage import data
from skimage.morphology import cmorph,disk
from skimage.filter.rank import _crank8, _crank16
from skimage.filter.rank import _crank16_percentiles


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def test_trivial_selem(self):
        # check that min, max and mean returns identity if structuring element contains only central pixel

        a = np.zeros((5,5),dtype='uint8')
        a[2,2] = 255
        a[2,3] = 128
        a[1,2] = 16
        elem = np.asarray([[0,0,0],[0,1,0],[0,0,0]],dtype='uint8')
        f = _crank8.mean(image=a,selem = elem,shift_x=0,shift_y=0)
        np.testing.assert_array_equal(a,f)
        f = _crank8.minimum(image=a,selem = elem,shift_x=0,shift_y=0)
        np.testing.assert_array_equal(a,f)
        f = _crank8.maximum(image=a,selem = elem,shift_x=0,shift_y=0)
        np.testing.assert_array_equal(a,f)

    def test_smallest_selem(self):
        # check that min, max and mean returns identity if structuring element contains only central pixel

        a = np.zeros((5,5),dtype='uint8')
        a[2,2] = 255
        a[2,3] = 128
        a[1,2] = 16
        elem = np.asarray([[1]],dtype='uint8')
        f = _crank8.mean(image=a,selem = elem,shift_x=0,shift_y=0)
        np.testing.assert_array_equal(a,f)
        f = _crank8.minimum(image=a,selem = elem,shift_x=0,shift_y=0)
        np.testing.assert_array_equal(a,f)
        f = _crank8.maximum(image=a,selem = elem,shift_x=0,shift_y=0)
        np.testing.assert_array_equal(a,f)



if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)
