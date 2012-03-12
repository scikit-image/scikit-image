"""
===============================================================================
Haar Salient Point Detector
===============================================================================

The Haar salient point detector detects "interest points" using a 2D Haar
wavelet decomposition.

In this example we can see that the types and number of features found is similar
between Haar and Harris detectors. The Haar detector does seem to have some possible
'false' features, compared to the Harris detector.

"""

from matplotlib import pyplot as plt

from skimage import data, img_as_float
from skimage.feature import harris, haar

def plot_points(image, filtered_coords, title):
    """ plots features found in image"""
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.plot([p[1] for p in filtered_coords],
             [p[0] for p in filtered_coords],
             'b.')
    plt.axis('off')
    plt.title(title)
    plt.draw()


im = img_as_float(data.camera())

haar_filtered_coords = haar(im, levels=5, threshold=0.2)
plot_points(im, haar_filtered_coords, 'Haar')

harris_filtered_coords = harris(im)
plot_points(im, harris_filtered_coords, 'Harris')

plt.show()
