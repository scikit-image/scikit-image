# -*- coding: utf-8 -*-
"""
=====================
Deconvolution of Lena
=====================

In this example, we deconvolve a noisy version of Lena using wiener
and unsupervised wiener algorithms. This algorithms are known to not
have the best respect of sharp edge in the image.

Wiener filter
-------------

This is simply the inverse filter based on the PSF, the prior
regularisation (penalisation of high frequency) and the tradeoff
between the data and prior adequacy. The regularization parameter must
be hand tuned.

Unsupervised wiener
-------------------

This algorithm has a self tuned regularisation parameters based on
data learning. This is not common and based on the following publication

.. [1] François Orieux, Jean-François Giovannelli, and Thomas
       Rodet, "Bayesian estimation of regularization and point
       spread function parameters for Wiener-Hunt deconvolution",
       J. Opt. Soc. Am. A 27, 1593-1607 (2010)
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import color, data, deconvolution

lena = color.rgb2gray(data.lena())
from scipy.signal import convolve2d as conv2
psf = np.ones((5, 5))
lena = conv2(lena, psf, 'same')
lena += 0.1 * lena.std() * np.random.standard_normal(lena.shape)

deconvolued, _ = deconvolution.unsupervised_wiener(lena, psf)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

plt.gray()

ax[0].imshow(lena)
ax[0].axis('off')
ax[0].set_title('Data')

ax[1].imshow(deconvolued)
ax[1].axis('off')
ax[1].set_title('Deconvolution')

fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)

plt.show()
