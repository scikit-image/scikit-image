"""
===========================================================
Multi-Block Local Binary Pattern for texture classification
===========================================================

In this example, we will see how to compute the multi-block
local binary pattern at a specified image and how to visualize it.

The features are calculated in a way similar to local binary
patterns, except that summed up pixel values
rather than pixel values are used.

`MB-LBP` is an extension of LBP that can be computed on any
scale in a constant time using integral image. It consists of
`9` equal-sized rectangles. They are used to compute a feature.
Sum of pixels' intensity values in each of them are compared
to the central rectangle and depending on comparison result,
the feature descriptor is computed.

We will start with a simple image that we will generate
to show how the `MB-LBP` works. We will create a `(9, 9)`
rectangle with and divide it into `9` blocks. After this
we will apply `MB-LBP` on it.


"""
from __future__ import print_function
from skimage.feature import multiblock_local_binary_pattern
import numpy as np
from skimage.util import img_as_float
from skimage.transform import integral_image

# Create dummy matrix where first and fifth
# rectangles have greater value than the central one
# Therefore, the following bits should be 1.
test_img = np.zeros((9, 9), dtype='uint8')
test_img[3:6, 3:6] = 1
test_img[:3, :3] = 50
test_img[6:, 6:] = 50

# MB-LBP is filled in reverse order.
# So the first and fifth bits from the end should
# be filled.
correct_answer = 0b10001000

# The function accepts the float images.
# Also it has to be C-contiguous.
test_img = img_as_float(test_img)
int_img = integral_image(test_img)

lbp_code = multiblock_local_binary_pattern(int_img, 0, 0, 3, 3)

print(lbp_code == correct_answer)

"""
Now let's apply the operator to a real image and see how the visualization works.
"""
from skimage import data
from matplotlib import pyplot as plt
from skimage.feature import draw_multiblock_lbp

test_img = data.coins()

test_img = img_as_float(test_img)
int_img = integral_image(test_img)

lbp_code = multiblock_local_binary_pattern(int_img, 0, 0, 90, 90)

img = draw_multiblock_lbp(test_img, 0, 0, 90, 90,
                          lbp_code=lbp_code, alpha=0.5)


plt.imshow(img, interpolation='nearest')

"""
.. image:: PLOT2RST.current_figure

On the above plot we see the result of computing a `MB-LBP` and visualization
of the computed feature. The rectangles that have less intensity than the central
rectangle are marked with cyan color. The ones that have bigger intensity values
are marked with white color. The central rectangle is left untouched.
"""
