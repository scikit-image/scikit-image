"""
===========================
Structural similarity index
===========================

When comparing images, the mean squared error (MSE)--while simple to
implement--is not highly indicative of perceived similarity.  Structural
similarity aims to address this shortcoming by taking texture into account
[1]_, [2]_.

The example shows two modifications of the input image, each with the same MSE,
but with very different mean structural similarity indices.

.. [1] Zhou Wang; Bovik, A.C.; ,"Mean squared error: Love it or leave it? A new
       look at Signal Fidelity Measures," Signal Processing Magazine, IEEE,
       vol. 26, no. 1, pp. 98-117, Jan. 2009.

.. [2] Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality
       assessment: From error visibility to structural similarity," IEEE
       Transactions on Image Processing, vol. 13, no. 4, pp. 600-612,
       Apr. 2004.

"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.measure import structural_similarity as ssim


matplotlib.rcParams['font.size'] = 9


img = img_as_float(data.camera())
rows, cols = img.shape

noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
noise[np.random.random(size=noise.shape) > 0.5] *= -1


def mse(x, y):
    return np.linalg.norm(x - y)

img_noise = img + noise
img_const = img + abs(noise)

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))

mse_none = mse(img, img)
ssim_none = ssim(img, img, dynamic_range=img.max() - img.min())

mse_noise = mse(img, img_noise)
ssim_noise = ssim(img, img_noise, dynamic_range=img_const.max() - img_const.min())

mse_const = mse(img, img_const)
ssim_const = ssim(img, img_const, dynamic_range=img_noise.max() - img_noise.min())

label = 'MSE: %2.f, SSIM: %.2f'

ax0.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
ax0.set_xlabel(label % (mse_none, ssim_none))
ax0.set_title('Original image')

ax1.imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
ax1.set_xlabel(label % (mse_noise, ssim_noise))
ax1.set_title('Image with noise')

ax2.imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
ax2.set_xlabel(label % (mse_const, ssim_const))
ax2.set_title('Image plus constant')

plt.show()
