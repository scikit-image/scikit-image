# Authors: Brian Holt
#
# License: BSD

import numpy as np
import scipy as sp
from scipy import ndimage

from numpy.testing import assert_raises

from scikits.image.feature import histogram_of_oriented_gradients  

def test_histogram_of_oriented_gradients():
    img = sp.lena().astype(np.int8) 
    
    fd, hog_image = histogram_of_oriented_gradients(img, 
                                                    n_orientations=9, 
                                                    ppc=(8,8), 
                                                    cpb=(1,1), 
                                                    visualise=False)
    assert len(fd) == 9 * (512//8) ** 2
    
if __name__ == '__main__':
    import nose
    nose.runmodule()
