import unittest

import numpy as np
from time import time
import matplotlib.pyplot as plt
from skimage import data

from tools import log_timing,init_logger

import crank
import crank16
import crank_percentiles
import crank16_percentiles
import crank16_bilateral
from cmorph import dilate


@log_timing
def c_max(image,selem):
    return crank.maximum(image=image,selem = selem)

@log_timing
def cm_max(image,selem):
    return dilate(image=image,selem = selem)

def compare():
    """comparison between
    - Cython maximum rankfilter implementation
    - weaves maximum rankfilter implementation
    - cmorph.dilate cython implementation
    on increasing structuring element size and increasing image size
    """
    a = (np.random.random((500,500))*256).astype('uint8')

    rec = []
    for r in range(1,20,1):
        elem = np.ones((r,r),dtype='uint8')
        #        elem = (np.random.random((r,r))>.5).astype('uint8')
        (rc,ms_rc) = c_max(a,elem)
        (rcm,ms_rcm) = cm_max(a,elem)
        rec.append((ms_rc,ms_rw,ms_rcm))
        assert  (rc==rcm).all()

    rec = np.asarray(rec)

    plt.plot(rec)
    plt.legend(['sliding cython','sliding weaves','cmorph'])
    plt.figure()
    plt.imshow(np.hstack((rc,rw,rcm)))

    r = 9
    elem = np.ones((r,r),dtype='uint8')

    rec = []
    for s in range(100,1000,100):
        a = (np.random.random((s,s))*256).astype('uint8')
        (rc,ms_rc) = c_max(a,elem)
        (rcm,ms_rcm) = cm_max(a,elem)
        rec.append((ms_rc,ms_rw,ms_rcm))
        assert  (rc==rcm).all()

    rec = np.asarray(rec)

    plt.figure()
    plt.plot(rec)
    plt.legend(['sliding cython','sliding weaves','cmorph'])
    plt.figure()
    plt.imshow(np.hstack((rc,rcm)))

    plt.show()
class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def test_random_sizes(self):
        # make sure the size is not a problem
        niter = 10
        elem = np.asarray([[1,1,1],[1,1,1],[1,1,1]],dtype='uint8')
        for m,n in np.random.random_integers(1,100,size=(10,2)):
            a8 = np.ones((m,n),dtype='uint8')
            r = crank.mean(image=a8,selem = elem,shift_x=0,shift_y=0)
            self.assertTrue(a8.shape == r.shape)
            r = crank.mean(image=a8,selem = elem,shift_x=+1,shift_y=+1)
            self.assertTrue(a8.shape == r.shape)

        for m,n in np.random.random_integers(1,100,size=(10,2)):
            a16 = np.ones((m,n),dtype='uint16')
            r = crank16.mean(image=a16,selem = elem,shift_x=0,shift_y=0)
            self.assertTrue(a16.shape == r.shape)
            r = crank16.mean(image=a16,selem = elem,shift_x=+1,shift_y=+1)
            self.assertTrue(a16.shape == r.shape)

        for m,n in np.random.random_integers(1,100,size=(10,2)):
            a16 = np.ones((m,n),dtype='uint16')
            r = crank16_percentiles.mean(image=a16,selem = elem,shift_x=0,shift_y=0,p0=.1,p1=.9)
            self.assertTrue(a16.shape == r.shape)
            r = crank16_percentiles.mean(image=a16,selem = elem,shift_x=+1,shift_y=+1,p0=.1,p1=.9)
            self.assertTrue(a16.shape == r.shape)

    def test_compare_with_cmorph(self):
        #compare the result of maximum filter with dilate
        a = (np.random.random((500,500))*256).astype('uint8')

        for r in range(1,20,1):
            elem = np.ones((r,r),dtype='uint8')
            #        elem = (np.random.random((r,r))>.5).astype('uint8')
            rc = crank.maximum(image=a,selem = elem)
            cm = dilate(image=a,selem = elem)
            self.assertTrue((rc==cm).all())

    def test_bitdepth(self):
        elem = np.ones((3,3),dtype='uint8')
        a16 = np.ones((100,100),dtype='uint16')*255
        r = crank16_percentiles.mean(image=a16,selem = elem,shift_x=0,shift_y=0,p0=.1,p1=.9,bitdepth=8)
        a16 = np.ones((100,100),dtype='uint16')*255*2
        r = crank16_percentiles.mean(image=a16,selem = elem,shift_x=0,shift_y=0,p0=.1,p1=.9,bitdepth=9)
        a16 = np.ones((100,100),dtype='uint16')*255*4
        r = crank16_percentiles.mean(image=a16,selem = elem,shift_x=0,shift_y=0,p0=.1,p1=.9,bitdepth=10)
        a16 = np.ones((100,100),dtype='uint16')*255*8
        r = crank16_percentiles.mean(image=a16,selem = elem,shift_x=0,shift_y=0,p0=.1,p1=.9,bitdepth=11)
        a16 = np.ones((100,100),dtype='uint16')*255*16
        r = crank16_percentiles.mean(image=a16,selem = elem,shift_x=0,shift_y=0,p0=.1,p1=.9,bitdepth=12)

    def test_population(self):
        a = np.zeros((5,5),dtype='uint8')
        elem = np.ones((3,3),dtype='uint8')
        p = crank.pop(image=a,selem = elem)
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
        f = crank.maximum(image=a,selem = elem,shift_x=1,shift_y=1)
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
        f = crank16_percentiles.mean(image=a16,selem = elem,shift_x=0,shift_y=0,p0=.1,p1=.9,bitdepth=4)

if __name__ == '__main__':

    logger = init_logger('app.log')

#    unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)


#    compare()

#    a = (data.coins()).astype('uint8')
    a8 = (data.coins()).astype('uint8')
    a = (data.coins()).astype('uint16')*16
    selem = np.ones((20,20),dtype='uint8')
#    f1 = filter.soft_gradient(a,struct_elem = selem,bitDepth=8,infSup=[.1,.9])
#    f2 = crank16.bottomhat(a,selem = selem,bitdepth=12)
    f1 = crank_percentiles.mean(a8,selem = selem,p0=.1,p1=.9)
#    f2 = crank16_percentiles.mean(a,selem = selem,bitdepth=12,p0=.1,p1=.9)
    f2 = crank16_bilateral.mean(a,selem = selem,bitdepth=12,s0=500,s1=500)
#    plt.imshow(f2)
    plt.imshow(np.hstack((a,f2)))
    plt.colorbar()
    plt.show()


