"""
=========================
Blind Image Deconvolution
=========================

Normally, image deconvolution is based on prior knowledge of the
Point Spread Function (PSF) used to deconvolve the image.
However, _blind_ methods are available that estimate the PSF
from the image itself. This algorithm is based on the
Richardson Lucy (RL) deconvolution algorithm. In this case,
the RL algorithm is not only used for deconvolving the image,
but also for the PSF estimation. This process is iterative,
alternating between deconvolving the PSF and deconvolving the image.

The following example shows a centered cross that was convolved
with a gaussian kernel with ``sigma=2``. Thereafter, Poisson
shot noise was added. Using the convolved image as argument
in the blind image deconvolution function, the algorithm
is capable to recover to a large extent the original image
and a good guess for the PSF (Figure 1).

Figure 2 shows the iterative progress where the first images look
like artifacts, but after a given amount of iterations,
the cross and the PSF nicely show up. Figure 3 shows the absolute
difference between the original image and the deconvolved one
indicating that the optimal number of iterations is 33.

.. [1] William Hadley Richardson, "Bayesian-Based Iterative
       Method of Image Restoration",
       J. Opt. Soc. Am. A 27, 1593-1607 (1972), :DOI:`10.1364/JOSA.62.000055`

.. [2] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution

.. [3] Fish, D. A., A. M. Brinicombe, E. R. Pike, and J. G. Walker.
       "Blind deconvolution by means of the Richardson–Lucy algorithm."
       JOSA A 12, no. 1 (1995): 58-65. :DOI:`10.1364/JOSAA.12.000058`

       https://pdfs.semanticscholar.org/9e3f/a71e22caf358dbe873e9649f08c205d0c0c0.pdf
"""
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

from skimage.filters import gaussian
from skimage.restoration import richardson_lucy

# Initialize image that is to be recovered (cross)
noisy_image = np.zeros((100, 100), dtype=np.float32)
noisy_image[40:60, 45:55] = 1
noisy_image[45:55, 40:60] = 1

# Add some poisson photon shot noise
np.random.seed(0)
noisy_image += np.random.poisson(2.0, noisy_image.shape) / 255

# Create the PSF
psf_gaussian = np.zeros_like(noisy_image)
w, h = noisy_image.shape
psf_gaussian[w // 2, h // 2] = 1
psf_gaussian = gaussian(psf_gaussian, 2)

# Convolve image using PSF
noisy_image_conv = convolve2d(noisy_image, psf_gaussian, 'same')
iterations = 50

# Run blind deconvolution and try to recover the used PSF
reconstruction = richardson_lucy(noisy_image_conv,
                                 iterations=iterations,
                                 return_iterations=True)


plt.figure(figsize=(12, 14))

for i in range(iterations * 2):
    plt.subplot(10, 10, i + 1)
    plt.imshow(reconstruction[i // 2, i % 2], cmap='gray')
    plt.axis('off')

# Calculate residuals from reconstruction array
residuals = np.empty(reconstruction.shape[0])

for i in range(reconstruction.shape[0]):
    residuals[i] = (noisy_image - reconstruction[i, 0] ** 2).sum()

best_fit = np.argmin(residuals)

plt.figure()
plt.plot(residuals)
plt.scatter(best_fit, residuals[best_fit], marker='o', s=100, alpha=.5)
plt.ylabel('Residuals')
plt.xlabel('Iteration#')


plt.figure(figsize=(12, 6))
plt.subplot(151)
plt.imshow(noisy_image, cmap='gray')
plt.title('Source Image')

plt.subplot(152)
plt.imshow(psf_gaussian, cmap='gray')
plt.title('Source PSF')

plt.subplot(153)
plt.imshow(noisy_image_conv, cmap='gray')
plt.title('Convolved image')

plt.subplot(154)
plt.imshow(reconstruction[best_fit, 0], cmap='gray')
plt.title('Recovered image,\n iteration #{}'.format(best_fit + 1))

plt.subplot(155)
plt.imshow(reconstruction[best_fit, 1], cmap='gray')
plt.title('Recovered PSF,\n iteration #{}'.format(best_fit + 1))

plt.tight_layout()

plt.show()
