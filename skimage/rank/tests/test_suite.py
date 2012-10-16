import unittest

import numpy as np

from skimage.rank import _crank8,_crank8_percentiles
from skimage.rank import _crank16,_crank16_bilateral,_crank16_percentiles
from skimage.morphology import cmorph,disk
from skimage import data
from skimage import rank


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

    def test_compare_with_cmorph_dilate(self):
        #compare the result of maximum filter with dilate

        a = (np.random.random((500,500))*256).astype('uint8')

        for r in range(1,20,1):
            elem = np.ones((r,r),dtype='uint8')
            #        elem = (np.random.random((r,r))>.5).astype('uint8')
            rc = _crank8.maximum(image=a,selem = elem)
            cm = cmorph.dilate(image=a,selem = elem)
            self.assertTrue((rc==cm).all())

    def test_compare_with_cmorph_erode(self):
        #compare the result of maximum filter with erode

        a = (np.random.random((500,500))*256).astype('uint8')

        for r in range(1,20,1):
            elem = np.ones((r,r),dtype='uint8')
            #        elem = (np.random.random((r,r))>.5).astype('uint8')
            rc = _crank8.minimum(image=a,selem = elem)
            cm = cmorph.erode(image=a,selem = elem)
            self.assertTrue((rc==cm).all())

    def test_bitdepth(self):
        # test the different bit depth for rank16
        
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
        # check the number of valid pixels in the neighborhood

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
        # check the output for a custom structuring element

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

    def test_output(self):
        #check rank function with external OUT output array

        selem = disk(20)
        a = (np.random.random((500,500))*256).astype('uint8')
        out = np.zeros_like(a)
        f1 = rank.mean(a,selem,out=out)
        f2 = rank.mean(a,selem)
        np.testing.assert_array_equal(f1,f2)
        np.testing.assert_array_equal(out,f2)

    @unittest.expectedFailure
    def test_inplace_output(self):
        #rank filters are not supposed to filter inplace

        selem = disk(20)
        a = (np.random.random((500,500))*256).astype('uint8')
        out = a
        f = rank.mean(a,selem,out=out)
        np.testing.assert_array_equal(f,out)


    def test_compare_autolevels(self):
        # compare autolevel and percentile autolevel with p0=0.0 and p1=1.0
        # should returns the same arrays

        image = data.camera()

        selem = disk(20)
        loc_autolevel = rank.autolevel(image,selem=selem)
        loc_perc_autolevel = rank.percentile_autolevel(image,selem=selem,p0=.0,p1=1.)

        assert (loc_autolevel==loc_perc_autolevel).all()

    def test_compare_autolevels_16bit(self):
        # compare autolevel(16bit) and percentile autolevel(16bit) with p0=0.0 and p1=1.0
        # should returns the same arrays

        image = data.camera().astype(np.uint16)*4

        selem = disk(20)
        loc_autolevel = rank.autolevel(image,selem=selem)
        loc_perc_autolevel = rank.percentile_autolevel(image,selem=selem,p0=.0,p1=1.)

        assert (loc_autolevel==loc_perc_autolevel).all()

    def test_compare_8bit_vs_16bit(self):
        # filters applied on 8bit image ore 16bit image (having only real 8bit of dynamic)
        # should be identical

        i8 = data.camera()
        i16 = i8.astype(np.uint16)
        assert (i8==i16).all()

        methods = ['autolevel','bottomhat','equalize','gradient','maximum','mean'
            ,'meansubstraction','median','minimum','modal','morph_contr_enh','pop','threshold', 'tophat']

        for method in methods:
            func = eval('rank.%s'%method)
            f8 = func(i8,disk(3))
            f16 = func(i16,disk(3))
            assert (f8==f16).all()


if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)
