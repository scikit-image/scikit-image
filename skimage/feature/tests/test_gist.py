import numpy as np
import scipy

import skimage.data 
from skimage.feature import gist 

def test_gist():
    im = skimage.data.lena()
    descs = gist.gist(im)
    first_values = [0.01281233, 0.01326025, 0.02414293, 0.03138399, 0.00965376]
    assert len(descs) == 960
    np.testing.assert_almost_equal(descs[0:5], first_values)
    
    
if __name__ == '__main__':
    from numpy.testing import run_module_suite
    run_module_suite()
