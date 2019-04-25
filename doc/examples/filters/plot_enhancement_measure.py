"""
=========================================
Quantitative Measure of Image Enhancement
=========================================

In this example, we will quantify the effect of applying contrast enhancement
to an image using a measure called EME [1]_.

It is defined as a log of the ratio of local maximum to local minimum
within a sliding window of given size.
This function was defined to quantify improvement of the image after processing.

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import gaussian
from skimage.exposure import equalize_adapthist
from skimage.measure.simple_metrics import enhancement_measure


img_gray = data.camera()
img_rgb = data.astronaut()


def plotter(image_1: np.ndarray, image_2: np.ndarray):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 14))
    for idx, image in enumerate([image_1, image_2]):
        ax[idx].imshow(image)
        ax[idx].axis('off')
        ax[idx].set_title("EME = {:.2f} ".format(enhancement_measure(image, size=3)))
    plt.show()


#####################################################################
# We will create a function that takes an original image as `before`
# and applies a `transform` to it. We will use a gaussian filter to
# strongly blur the image and an adaptive histogram equalization as
# transforms.

def compare_eme(before: np.ndarray, transform, kwargs=None):
    after = transform(before, **kwargs)
    plotter(before, after)

compare_eme(before=img_gray, transform=gaussian, kwargs={'sigma':10})
compare_eme(before=img_rgb, transform=equalize_adapthist, kwargs={'nbins':256*3})

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
# .. [1] Agaian, Sos S., Karen Panetta, and Artyom M. Grigoryan.
# "A new measure of image enhancement."
# IASTED International Conference on Signal Processing
# & Communication. Citeseer, 2000,
# :DOI:10.1.1.35.4021
