"""
==========================================================
Apply a set of "Gabor" and "Morlet" filters to an picture
==========================================================

In this example, we show the difference between filtering an image with the
Gabor filter and the Morlet filter.


Morlet Filter
---------------------

Zero sum version of the Gabor filter.

"""
import numpy as np
import skimage
from skimage.filters import gabor_kernel
from skimage.filters import morlet_kernel
import matplotlib.pylab as plt
from skimage import data
from skimage.util import img_as_float
from scipy import ndimage as ndi

image = img_as_float(data.load('brick.png'))
image = image[0:64,0:64]


J = 4
L = 8
xi_psi = 3. / 4 * np.pi
sigma_xi = .8
slant = 4. / L

#show image
plt.figure(figsize=(16, 8))
plt.imshow(image)
plt.title('Original image')

# Generate a group of gabor filters and apply it to the brick image

plt.figure(figsize=(16, 8))
for j, scale in enumerate(2 ** np.arange(J)):
    for l, theta in enumerate(np.arange(L) / float(L) * np.pi):
        sigma = sigma_xi * scale
        xi = xi_psi / scale

        sigma_x = sigma
        sigma_y = sigma / slant
        freq = xi / (np.pi * 2)

        gabor = gabor_kernel(freq, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y)

        im_filtered = np.abs(ndi.convolve(image, gabor, mode='wrap'))

        plt.subplot(J, L, j * L + l + 1)
        plt.imshow(np.real(im_filtered), interpolation='nearest')

        plt.viridis()

plt.suptitle('Gabor (different scales and orientations)')
# Generate a group of morlet filters and apply it to the brick image

plt.figure(figsize=(16, 8))
for j, scale in enumerate(2 ** np.arange(J)):
    for l, theta in enumerate(np.arange(L) / float(L) * np.pi):
        sigma = sigma_xi * scale
        xi = xi_psi / scale

        sigma_x = sigma
        sigma_y = sigma / slant
        freq = xi / (np.pi * 2)

        morlet = morlet_kernel(freq, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y)

        im_filtered = np.abs(ndi.convolve(image, morlet, mode='wrap'))

        plt.subplot(J, L, j * L + l + 1)
        plt.imshow(np.real(im_filtered), interpolation='nearest')

        plt.viridis()

plt.suptitle('Morlet (different scales and orientations)')

plt.show()

print('The energy of the filtered image changes with the gabor fiter but not with the Gabor:')
im_filtered = np.abs(ndi.convolve(image, morlet, mode='wrap'))
print('[Morlet] energy:',im_filtered.sum())
im_filtered100 = np.abs(ndi.convolve(image+100, morlet, mode='wrap'))
print('[Morlet] energy (im+100):',im_filtered100.sum())

im_filtered = np.abs(ndi.convolve(image, gabor, mode='wrap'))
print('[Gabor] energy:',im_filtered.sum())
im_filtered100 = np.abs(ndi.convolve(image+100, gabor, mode='wrap'))
print('[Gabor] energy (im+100):',im_filtered100.sum())
