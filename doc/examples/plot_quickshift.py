import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage
#from skimage.data import lena
#from skimage.util import img_as_float
from skimage.segmentation import quickshift

from IPython.core.debugger import Tracer
tracer = Tracer()


def microstructure(l=256):
    """
    Synthetic binary data: binary microstructure with blobs.

    Parameters
    ----------

    l: int, optional
        linear size of the returned image
    """
    n = 5
    x, y = np.ogrid[0:l, 0:l]
    mask = np.zeros((l, l))
    generator = np.random.RandomState(1)
    points = l * generator.rand(2, n ** 2)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndimage.gaussian_filter(mask, sigma=l / (4. * n))
    return (mask > mask.mean()).astype(np.float)


#img = img_as_float(lena()[250:300, 250:300])
img = microstructure(l=50)
segments = quickshift(img.reshape(50, 50, 1))
segments = np.unique(segments, return_inverse=True)[1].reshape(50, 50)
intensities = np.bincount(segments.ravel(), img.ravel())
counts = np.bincount(segments.ravel())
intensities /= counts

plt.imshow(img, interpolation='nearest')
plt.figure()
plt.imshow(segments, interpolation='nearest')
plt.figure()
plt.imshow(intensities[segments], interpolation='nearest')
plt.show()
print("num segments: %d" % len(np.unique(segments)))
