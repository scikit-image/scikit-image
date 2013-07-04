import numpy as np
from matplotlib.pyplot import imshow as sh
plt.gray()

import _heap
import fmm
#import _inpaint
from skimage.io import imread as re

# Import the 20X20 image and mask
image = np.array(re('/Users/chintak/Pictures/images/t_im.jpg', as_grey=True))
mask = np.zeros(image.shape, np.uint8)
mask[15:19, 15: 19] = 1

# Generate `flag` and `u` matrices
flag = _heap.init_flag(mask)
u = _heap.init_u(flag)

# Initialize the heap array
heap = []
_heap.generate_heap(heap, flag, u)

epsilon = 5
output = fmm.fast_marching_method(image, flag, u, heap, negate=False,
                                  epsilon=epsilon)

sh(output)
pass
