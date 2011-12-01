import numpy as np
from numpy.testing import *

from skimage.measure import find_contours 

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

x,y = np.mgrid[-1:1:5j,-1:1:5j]
r = np.sqrt(x**2 + y**2)

def test_binary():
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

def test_float():
  contours = find_contours(r, 0.5)
  assert len(contours) == 1
  assert_array_equal(contours[0],
                    [[ 2.,  3.],
                     [ 1.,  2.],
                     [ 2.,  1.],
                     [ 3.,  2.],
                     [ 2.,  3.]])


    
if __name__ == '__main__':
    from numpy.testing import run_module_suite
    run_module_suite()
