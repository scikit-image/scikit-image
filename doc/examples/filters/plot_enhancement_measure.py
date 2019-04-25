"""
=========================================
Quantitative Measure of Image Enhancement
=========================================

In this example, we will measure the result of applying contrast enhancement
to the image. We will use a measure called EME [1]_.

It is defined as a log of the ratio of local maximum to local minimum
within a sliding window. The code allows the user to choose the window size.
This function was defined to quantify improvement of the image
after processing.

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import gaussian
from skimage.exposure import equalize_adapthist
from skimage.measure import simple_metrics

img_gray = data.camera()
img_rgb = data.astronaut()


def compare_side_by_side(before: np.ndarray, after: np.ndarray):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 14))
    for idx, image in enumerate([before, after]):
        ax[idx].imshow(image)
        ax[idx].axis('off')
        ax[idx].set_title("EME = {:.2f} ".format(
            simple_metrics.enhancement_measure(image, size=3)))
    plt.show()


#####################################################################
# We will create a function that takes an original image as `before`
# and applies a `transform` to it. We will use a gaussian filter to
# strongly blur the image and an adaptive histogram equalization as
# transforms.


compare_side_by_side(before=img_gray, after=gaussian(img_gray, sigma=10))
compare_side_by_side(before=img_rgb, after=equalize_adapthist(nbins=256 * 3))

############################################################################
# You can see that the greyscale camera image is very blurred due to
# strong gaussian filtering. This resulted in smoothing a lot of pixel
# differences and, as a result, to lower value of EME.
# On the other hand, the `astronaut` image was enhanced by adaptive
# histogram equalization with `256*3` bins. Histogram equalization
# brought out details and the EME value is higher than before the transform.

############################################################################
# References
# ----------
# .. [1] Sos S. Agaian, Karen Panetta, and Artyom M. Grigoryan.
# "A new measure of image enhancement."
# IASTED International Conference on Signal Processing
# & Communication. Citeseer, 2000,
# :DOI:10.1.1.35.4021
