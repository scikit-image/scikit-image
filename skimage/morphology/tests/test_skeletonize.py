import numpy as np
from scikits.image.morphology import skeletonize
import numpy.testing
from scikits.image.draw import draw
from scipy.ndimage import correlate
from scikits.image.io import imread
from scikits.image import data_dir
import os.path

class TestSkeletonize():
    def test_skeletonize_no_foreground(self):
        im = np.zeros((5,5))
        result = skeletonize(im)
        numpy.testing.assert_array_equal(result, np.zeros((5,5)))
    
    def test_skeletonize_wrong_dim1(self):
        im = np.zeros((5))
        numpy.testing.assert_raises(ValueError, skeletonize, im)    

    def test_skeletonize_wrong_dim2(self):
        im = np.zeros((5, 5, 5))
        numpy.testing.assert_raises(ValueError, skeletonize, im)    

    def test_skeletonize_not_binary(self):
        im = np.zeros((5, 5))
        im[0, 0] = 1
        im[0, 1] = 2
        numpy.testing.assert_raises(ValueError, skeletonize, im)  
        
    def test_skeletonize_unexpected_value(self):
        im = np.zeros((5, 5))
        im[0, 0] = 2
        numpy.testing.assert_raises(ValueError, skeletonize, im)          
        
    def test_skeletonize_all_foreground(self):
        im = np.ones((3,4))
        result = skeletonize(im)
    
    def test_skeletonize_single_point(self):
        im = np.zeros((5, 5), np.uint8)
        im[3, 3] = 1
        result = skeletonize(im)
        numpy.testing.assert_array_equal(result, im)
    
    def test_skeletonize_already_thinned(self):
        im = np.zeros((5, 5), np.uint8)
        im[3,1:-1] = 1
        im[2, -1] = 1
        im[4, 0] = 1
        result = skeletonize(im)
        numpy.testing.assert_array_equal(result, im)
    
    def test_skeletonize_output(self):
        im = imread(os.path.join(data_dir, "bw_text.png"), as_grey=True)
        
        # make black the foreground
        im = (im==0)
        result = skeletonize(im)
        
        expected = np.load(os.path.join(data_dir, "bw_text_skeleton.npy"))
        numpy.testing.assert_array_equal(result, expected)
        
    
    def test_skeletonize_num_neighbours(self):
        # an empty image
        image = np.zeros((300, 300))
        
        # foreground object 1
        image[10:-10, 10:100] = 1
        image[-100:-10, 10:-10] = 1
        image[10:-10, -100:-10] = 1
        
        # foreground object 2
        rs, cs = draw.bresenham(250, 150, 10, 280)
        for i in range(10): image[rs+i, cs] = 1 
        rs, cs = draw.bresenham(10, 150, 250, 280)
        for i in range(20): image[rs+i, cs] = 1 
        
        # foreground object 3
        ir, ic = np.indices(image.shape)
        circle1 = (ic - 135)**2 + (ir - 150)**2 < 30**2
        circle2 = (ic - 135)**2 + (ir - 150)**2 < 20**2
        image[circle1] = 1
        image[circle2] = 0
        result = skeletonize(image)
        
        # there should never be a 2x2 block of foreground pixels in a skeleton
        mask = np.array([[1,  1],
                         [1,  1]], np.uint8)        
        blocks = correlate(result, mask, mode='constant')
        assert not numpy.any(blocks == 4)
        

if __name__ == '__main__':
    np.testing.run_module_suite()
