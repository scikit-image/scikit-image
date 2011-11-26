import numpy as np
from numpy.testing import *

from skimage.find_contours import find_contours 

a = np.ones((8,8), dtype=np.float32)
a[1:-1, 1] = 0
a[1, 1:-1] = 0

## array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
##        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
##        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
##        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
##        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
##        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
##        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
##        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]], dtype=float32)


def test_find_contours():
  contours = find_contours(a, 0.5)
  assert len(contours) == 1
  assert_array_equal(contours[0],
                     [[ 6. ,  1.5],
                      [ 5. ,  1.5],
                      [ 4. ,  1.5],
                      [ 3. ,  1.5],
                      [ 2. ,  1.5],
                      [ 1.5,  2. ],
                      [ 1.5,  3. ],
                      [ 1.5,  4. ],
                      [ 1.5,  5. ],
                      [ 1.5,  6. ],
                      [ 1. ,  6.5],
                      [ 0.5,  6. ],
                      [ 0.5,  5. ],
                      [ 0.5,  4. ],
                      [ 0.5,  3. ],
                      [ 0.5,  2. ],
                      [ 0.5,  1. ],
                      [ 1. ,  0.5],
                      [ 2. ,  0.5],
                      [ 3. ,  0.5],
                      [ 4. ,  0.5],
                      [ 5. ,  0.5],
                      [ 6. ,  0.5],
                      [ 6.5,  1. ],
                      [ 6. ,  1.5]])


    
if __name__ == '__main__':
    from numpy.testing import run_module_suite
    run_module_suite()
