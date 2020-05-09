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

# Recover every steps from the deconvolution
evolution = np.empty((iterations, 2,) + noisy_image.shape)


def cb(im_deconv, psf, it):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    evolution[it, 0] = im_deconv
    evolution[it, 1] = psf


# Run blind deconvolution and try to recover the used PSF
reconstruction, psf = richardson_lucy(noisy_image_conv,
                                      iterations=iterations,
                                      return_iterations=True,
                                      iter_callback=cb)

# Calculate residuals from reconstruction array
residuals = np.empty(evolution.shape[0])

for i in range(evolution.shape[0]):
    residuals[i] = (np.abs(noisy_image - evolution[i, 0]**2)).sum()

best_fit = np.argmin(residuals)


fig, ax = plt.subplots(ncols=6, figsize=(12, 6))

ax[0].imshow(noisy_image, cmap='gray')
ax[0].set_title('Source Image')

ax[1].imshow(reconstruction, cmap='gray')
ax[1].set_title('Result')

ax[2].imshow(psf_gaussian, cmap='gray')
ax[2].set_title('Source PSF')

ax[3].imshow(noisy_image_conv, cmap='gray')
ax[3].set_title('Convolved image')

ax[4].imshow(evolution[best_fit, 0], cmap='gray')
ax[4].set_title('Recovered image,\n iteration #{}'.format(best_fit + 1))

ax[5].imshow(evolution[best_fit, 1], cmap='gray')
ax[5].set_title('Recovered PSF,\n iteration #{}'.format(best_fit + 1))

plt.tight_layout()

plt.show()

###################################################
# Figure 2 shows the iterative progress where the first images look
# like artifacts, but after a given amount of iterations,
# the cross and the PSF nicely show up. Figure 3 shows the absolute
# difference between the original image and the deconvolved one
# indicating that the optimal number of iterations is 33.


plt.figure(figsize=(12, 14))

for i in range(iterations * 2):
    plt.subplot(10, 10, i + 1)
    plt.imshow(evolution[i // 2, i % 2], cmap='gray')
    plt.axis('off')
plt.show()

plt.figure()
plt.plot(residuals)
plt.scatter(best_fit, residuals[best_fit], marker='o', s=100, alpha=.5)
plt.ylabel('Residuals')
plt.xlabel('Iteration#')
plt.show()
