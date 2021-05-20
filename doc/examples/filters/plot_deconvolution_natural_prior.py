"""
==================================
Image Deconvolution Natural Priors
==================================
In this example, we deconvolve an image using deconvolution with natural
priors, e.g Gauss, solved in the frequency domain with a new
deconvolution algorithm ([1]_, [2]_, [3]_).

The algorithm is based on a PSF (Point Spread Function),
where PSF is described as the impulse response of the
optical system. The blurred image is sharpened with a smoothness weight,
which defines the weight of the Gauss prior during deconvolution .

.. [1] http://groups.csail.mit.edu/graphics/CodedAperture/
.. [2] Levin, A., Fergus, R., Durand, F., & Freeman, W. T. (2007).
       Deconvolution using natural image priors.
       Massachusetts Institute of Technology,
       Computer Science and Artificial Intelligence Laboratory, 3.
.. [3] Levin, A., Fergus, R., Durand, F., & Freeman, W. T. (2007).
       Image and depth from a conventional camera with a coded aperture.
       ACM transactions on graphics (TOG), 26(3), 70-es.
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration

astro = color.rgb2gray(data.astronaut())

psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
# Add Noise to Image
astro_noisy = astro.copy()
astro_noisy += (np.random.poisson(lam=25, size=astro.shape) - 10) / 255.

# Restore Image using Richardson-Lucy algorithm
deconvolved_DNP = restoration.gaussian_natural_prior(astro_noisy, psf)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
plt.gray()

for a in (ax[0], ax[1], ax[2]):
    a.axis('off')

ax[0].imshow(astro)
ax[0].set_title('Original Data')

ax[1].imshow(astro_noisy)
ax[1].set_title('Noisy data')

ax[2].imshow(deconvolved_DNP, vmin=astro_noisy.min(), vmax=astro_noisy.max())
ax[2].set_title('Restoration using\nnatural Gauss prior')


fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)
plt.show()
