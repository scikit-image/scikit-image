#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Image Deconvolution
=====================
<<<<<<< HEAD
In this example, we deconvolve an image using Richardson-Lucy
deconvolution algorithm ([1]_, [2]_).

The algorithm is based on a PSF (Point Spread Function),
where PSF is described as the impulse response of the
optical system. The blurred image is sharpened through a number of
iterations, which needs to be hand-tuned.

.. [1] William Hadley Richardson, "Bayesian-Based Iterative
       Method of Image Restoration",
       J. Opt. Soc. Am. A 27, 1593-1607 (1972), DOI:10.1364/JOSA.62.000055
=======

In this example, Wiener, unsupervised Wiener, and Richardson-Lucy
algorithms are used to deconvolve an image.

Wiener and unsupervised Wiener filters
--------------------------------------
First, a noisy version of an image is deconvolved using Wiener
and unsupervised Wiener algorithms. These algorithms are based on
<<<<<<< HEAD
linear models that can't restore sharp edges as much as non-linear
methods (like TV restoration), but are much faster.
>>>>>>> Merging deconvolution examples.
=======
linear models that cannot restore sharp edges as much as non-linear
methods (like TV restoration), but are faster.
>>>>>>> Answering requests.

The Wiener deconvolution algorithm is an inverse filter based on an
impulse response function (also called point-spread function, or PSF), a
filter to penalize the restored image frequencies (regularization), and
a parameter which tunes the balance between the data and the
regularization.

The unsupervised Wiener algorithm has self-tuned regularization
parameters based on the Gibbs sampling algorithm. The iterative Gibbs
sampler draws alternatively samples from posterior conditional law of
the image, the noise power and the image frequency power.

.. [1] François Orieux, Jean-François Giovannelli, and Thomas
       Rodet, "Bayesian estimation of regularization and point
       spread function parameters for Wiener-Hunt deconvolution",
       J. Opt. Soc. Am. A 27, 1593-1607 (2010),
       DOI:10.1364/JOSAA.27.001593
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration

astro = color.rgb2gray(data.astronaut())

# Add noise to image.
psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
astro += 0.1 * astro.std() * np.random.standard_normal(astro.shape)

# Restore image using unsupervised Wiener algorithm.
deconvolved, _ = restoration.unsupervised_wiener(astro, psf)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                       sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})

plt.gray()

ax[0].imshow(astro, vmin=deconvolved.min(), vmax=deconvolved.max())
ax[0].axis('off')
ax[0].set_title('Data')

ax[1].imshow(deconvolved)
ax[1].axis('off')
ax[1].set_title('Self tuned restoration')

fig.tight_layout()

plt.show()

"""
Richardson-Lucy filter
----------------------
Now an image is deconvolved using the Richardson-Lucy algorithm.

The algorithm is also based on a PSF, the impulse response of the
optical system. The blurred image is sharpened through a number of
iterations, which needs to be hand-tuned, as in Wiener filters.

.. [1] William Hadley Richardson, "Bayesian-Based Iterative Method of
       Image Restoration", J. Opt. Soc. Am. A 27, 1593-1607 (1972),
       DOI:10.1364/JOSA.62.000055

.. [2] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
"""

astro = color.rgb2gray(data.astronaut())

psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
<<<<<<< HEAD
# Add Noise to Image
=======

# Add noise to image.
>>>>>>> Merging deconvolution examples.
astro_noisy = astro.copy()
astro_noisy += (np.random.poisson(lam=25, size=astro.shape) - 10) / 255.

# Restore image using Richardson-Lucy algorithm.
deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, iterations=30)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
plt.gray()

for a in (ax[0], ax[1], ax[2]):
    a.axis('off')

ax[0].imshow(astro)
ax[0].set_title('Original Data')

ax[1].imshow(astro_noisy)
ax[1].set_title('Noisy data')

ax[2].imshow(deconvolved_RL,
             vmin=astro_noisy.min(),
             vmax=astro_noisy.max())
ax[2].set_title('Restoration using\nRichardson-Lucy')


fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05,
                    left=0, right=1)
plt.show()
