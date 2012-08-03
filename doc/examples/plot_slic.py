"""
"""
print __doc__

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import lena
from skimage.segmentation import slic, visualize_boundaries
from skimage.util import img_as_float

img = img_as_float(lena()).copy("C")
segments = slic(img, ratio=10.0, n_segments=1000)

print("number of segments: %d" % len(np.unique(segments)))

boundaries_mine = visualize_boundaries(img, segments)
plt.imshow(boundaries_mine)
plt.axis("off")
plt.show()
