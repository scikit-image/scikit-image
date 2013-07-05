import numpy as np
from matplotlib.pyplot import imshow as sh
plt.gray()

import _heap
import fmm
#import _inpaint
# from skimage.io import imread as re
# from skimage.morphology import dilation, square

# Import the 20X20 image and mask
# image=np.array(re('/Users/chintak/Pictures/images/t_im10.jpg', as_grey=True))
# image[3:6, 3:6] = 0

image = np.array([[33, 24, 27, 36, 27, 28, 36, 41, 32, 42],
                  [49, 42, 39, 54, 38, 44, 44, 49, 50, 58],
                  [69, 68, 65, 66, 45, 50, 53, 55, 57, 63],
                  [85, 81, 71, 0, 0, 0, 57, 63, 57, 59],
                  [87, 93, 99, 0, 0, 0, 78, 75, 65, 63],
                  [110, 103, 99, 0, 0, 0, 104, 106, 74, 69],
                  [109, 120, 151, 126, 132, 115, 122, 90, 76, 68],
                  [137, 121, 120, 131, 131, 127, 113, 94, 76, 67],
                  [110, 115, 122, 124, 120, 110, 98, 89, 87, 68],
                  [102, 109, 117, 123, 122, 114, 104, 97, 92, 72]],
                 dtype=np.uint8)
mask = np.zeros(image.shape, np.uint8)
mask[3:6, 3:6] = 1
epsilon = 3
con = image.copy()

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
# pass
plt.subplot(1, 2, 1), sh(con)
plt.subplot(1, 2, 2), sh(output)
plt.show()
