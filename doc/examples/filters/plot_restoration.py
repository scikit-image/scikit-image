# -*- coding: utf-8 -*-
"""
=====================
Image Deconvolution
=====================

In this example, we deconvolve a noisy version of an image using
Richardson–Lucy, Wiener, and unsupervised Wiener deconvolution algorithms.
This algorithms are based on linear models that can't restore sharp edge as
much as non-linear methods (like TV restoration) but are much faster.

Richardson–Lucy deconvolution
-------------
The inverse filter based on the PSF (Point Spread Function).
Regularisation is tuned through the number of iterations, which must
be hand tuned. The method assumes Poisson noise and is thus best
applied to data with e.g. photon noise (such as from photo diodes).

Wiener deconvolution
-------------
Another inverse filter based on the PSF (Point Spread Function),
the prior regularisation (penalisation of high frequency) and the
tradeoff between the data and prior adequacy. The regularization
parameter must be hand tuned.

Unsupervised Wiener
-------------------
This algorithm has a self-tuned regularisation parameters based on
data learning. This is not common and based on the following
publication. The algorithm is based on a iterative Gibbs sampler that
draw alternatively samples of posterior conditionnal law of the image,
the noise power and the image frequency power.

.. [1] François Orieux, Jean-François Giovannelli, and Thomas
       Rodet, "Bayesian estimation of regularization and point
       spread function parameters for Wiener-Hunt deconvolution",
       J. Opt. Soc. Am. A 27, 1593-1607 (2010)

Example with Poisson noise:
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import color, data, restoration

astro = color.rgb2gray(data.astronaut())

from scipy.signal import convolve2d as conv2
psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
astro_pois = astro.copy()
astro_pois += (np.random.poisson(lam=25, size=astro.shape)-10)/255.

deconvolved_RL = restoration.richardson_lucy(astro_pois, psf, iterations=20)
deconvolved_W = restoration.wiener(astro_pois, psf, balance=0.01)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
plt.gray()

ax[0].imshow(astro_pois)
ax[0].axis('off')
ax[0].set_title('Poisson noisy data')

ax[1].imshow(deconvolved_RL, vmin=astro_pois.min(), vmax=astro_pois.max())
ax[1].axis('off')
ax[1].set_title('Richardson Lucy')

ax[2].imshow(deconvolved_W, vmin=astro_pois.min(), vmax=astro_pois.max())
ax[2].axis('off')
ax[2].set_title('Wiener')
fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)

"""
.. image:: PLOT2RST.current_figure

With Gaussian noise:
"""
astro_std = astro.copy()
astro_std += 0.1 * astro.std() * np.random.standard_normal(astro.shape)

deconvolved_RL = restoration.richardson_lucy(astro_std, psf, iterations=20)
deconvolved_W = restoration.wiener(astro_std, psf, balance=0.01)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))

ax[0].imshow(astro_std)
ax[0].axis('off')
ax[0].set_title('Guassian noisy data')

ax[1].imshow(deconvolved_RL, vmin=astro_std.min(), vmax=astro_std.max())
ax[1].axis('off')
ax[1].set_title('Richardson Lucy')

ax[2].imshow(deconvolved_W, vmin=astro_std.min(), vmax=astro_std.max())
ax[2].axis('off')
ax[2].set_title('Wiener')

fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)

"""
.. image:: PLOT2RST.current_figure

Lastly, Unsupervised Wiener for which manual tuning of parameter
is not needed:
"""
deconvolved, _ = restoration.unsupervised_wiener(astro_std, psf)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

ax[0].imshow(astro_std)
ax[0].axis('off')
ax[0].set_title('Data')

ax[1].imshow(deconvolved, vmin=astro_std.min(), vmax=astro_std.max())
ax[1].axis('off')
ax[1].set_title('Self tuned restoration')

fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)
"""
.. image:: PLOT2RST.current_figure
"""

plt.show()
