import matplotlib.pyplot as plt
import numpy as np

from skimage.data import lena
from skimage.segmentation import felzenszwalb_segmentation

img = lena()
segments = felzenszwalb_segmentation(img, k=1000)
plt.imshow(img)
plt.figure()
plt.imshow(segments)
plt.show()
print("num segments: %d" % len(np.unique(segments)))
