import numpy as np
import scipy

from scikits.image.feature import hog 

def test_histogram_of_oriented_gradients():
    img = scipy.lena().astype(np.int8) 
    
    fd = hog(img, n_orientations=9, pixels_per_cell=(8, 8), 
             cells_per_block=(1, 1))
    assert len(fd) == 9 * (512//8) ** 2
    
if __name__ == '__main__':
    from numpy.testing import run_module_suite
    run_module_suite()
