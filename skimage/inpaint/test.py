import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow as sh
plt.gray()

import _heap
import fmm
#import _inpaint
from skimage.io import imread as re
# from skimage.morphology import dilation, square

# Import the 20X20 image and mask
image = np.array(re('/Users/chintak/Pictures/images/t_im10.jpg', as_grey=True))
image[3:6, 3:6] = 0
mask = np.zeros(image.shape, np.uint8)
mask[3:6, 3:6] = 1
epsilon = 3

sh(image)
plt.show()
# Generate `flag` and `u` matrices
flag = _heap.init_flag(mask)

# outside = dilation(mask, square(2 * epsilon + 1))
# outside_band = np.logical_xor(outside, mask).astype(np.uint8)
# out_flag = _heap.init_flag(outside_band)
u = _heap.init_u(flag)

# out_heap = []
# _heap.generate_heap(out_heap, out_flag, u)
# u = fmm.fast_marching_method(outside_band, out_flag, u, out_heap,negate=True)

# Initialize the heap array
heap = []
_heap.generate_heap(heap, flag, u)

output = fmm.fast_marching_method(image, flag, u, heap, negate=False,
                                  epsilon=epsilon)

sh(output)
plt.show()
