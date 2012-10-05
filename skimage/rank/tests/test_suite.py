import unittest

import numpy as np

from skimage.rank import _crank8,_crank8_percentiles
from skimage.rank import _crank16,_crank16_bilateral,_crank16_percentiles
from skimage.morphology import cmorph

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def test_random_sizes(self):
        # make sure the size is not a problem
        niter = 10
        elem = np.asarray([[1,1,1],[1,1,1],[1,1,1]],dtype='uint8')
        for m,n in np.random.random_integers(1,100,size=(10,2)):
            a8 = np.ones((m,n),dtype='uint8')
            r = _crank8.mean(image=a8,selem = elem,shift_x=0,shift_y=0)
            self.assertTrue(a8.shape == r.shape)
            r = _crank8.mean(image=a8,selem = elem,shift_x=+1,shift_y=+1)
            self.assertTrue(a8.shape == r.shape)

        for m,n in np.random.random_integers(1,100,size=(10,2)):
            a16 = np.ones((m,n),dtype='uint16')
            r = _crank16.mean(image=a16,selem = elem,shift_x=0,shift_y=0)
            self.assertTrue(a16.shape == r.shape)
            r = _crank16.mean(image=a16,selem = elem,shift_x=+1,shift_y=+1)
            self.assertTrue(a16.shape == r.shape)

        for m,n in np.random.random_integers(1,100,size=(10,2)):
            a16 = np.ones((m,n),dtype='uint16')
            r = _crank16_percentiles.mean(image=a16,selem = elem,shift_x=0,shift_y=0,p0=.1,p1=.9)
            self.assertTrue(a16.shape == r.shape)
            r = _crank16_percentiles.mean(image=a16,selem = elem,shift_x=+1,shift_y=+1,p0=.1,p1=.9)
            self.assertTrue(a16.shape == r.shape)

    def test_compare_with_cmorph(self):
        #compare the result of maximum filter with dilate
        a = (np.random.random((500,500))*256).astype('uint8')

        for r in range(1,20,1):
            elem = np.ones((r,r),dtype='uint8')
            #        elem = (np.random.random((r,r))>.5).astype('uint8')
            rc = _crank8.maximum(image=a,selem = elem)
            cm = cmorph.dilate(image=a,selem = elem)
            self.assertTrue((rc==cm).all())

    def test_bitdepth(self):
        elem = np.ones((3,3),dtype='uint8')
        a16 = np.ones((100,100),dtype='uint16')*255
        r = _crank16_percentiles.mean(image=a16,selem = elem,shift_x=0,shift_y=0,p0=.1,p1=.9,bitdepth=8)
        a16 = np.ones((100,100),dtype='uint16')*255*2
        r = _crank16_percentiles.mean(image=a16,selem = elem,shift_x=0,shift_y=0,p0=.1,p1=.9,bitdepth=9)
        a16 = np.ones((100,100),dtype='uint16')*255*4
        r = _crank16_percentiles.mean(image=a16,selem = elem,shift_x=0,shift_y=0,p0=.1,p1=.9,bitdepth=10)
        a16 = np.ones((100,100),dtype='uint16')*255*8
        r = _crank16_percentiles.mean(image=a16,selem = elem,shift_x=0,shift_y=0,p0=.1,p1=.9,bitdepth=11)
        a16 = np.ones((100,100),dtype='uint16')*255*16
        r = _crank16_percentiles.mean(image=a16,selem = elem,shift_x=0,shift_y=0,p0=.1,p1=.9,bitdepth=12)

    def test_population(self):
        a = np.zeros((5,5),dtype='uint8')
        elem = np.ones((3,3),dtype='uint8')
        p = _crank8.pop(image=a,selem = elem)
        r = np.asarray([[4, 6, 6, 6, 4],
            [6, 9, 9, 9, 6],
            [6, 9, 9, 9, 6],
            [6, 9, 9, 9, 6],
            [4, 6, 6, 6, 4]])
        np.testing.assert_array_equal(r,p)

    def test_structuring_element(self):
        a = np.zeros((6,6),dtype='uint8')
        a[2,2] = 255
        elem = np.asarray([[1,1,0],[1,1,1],[0,0,1]],dtype='uint8')
        f = _crank8.maximum(image=a,selem = elem,shift_x=1,shift_y=1)
        r = np.asarray([[  0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0],
            [  0,   0, 255,   0,   0,   0],
            [  0,   0, 255, 255, 255,   0],
            [  0,   0,   0, 255, 255,   0],
            [  0,   0,   0,   0,   0,   0]])
        np.testing.assert_array_equal(r,f)


    @unittest.expectedFailure
    def test_fail_on_bitdepth(self):
        # should fail because data bitdepth is too high for the function
        a16 = np.ones((100,100),dtype='uint16')*255
        elem = np.ones((3,3),dtype='uint8')
        f = _crank16_percentiles.mean(image=a16,selem = elem,shift_x=0,shift_y=0,p0=.1,p1=.9,bitdepth=4)

if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)
