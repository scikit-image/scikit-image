"""
=============
Guided filter
=============

The guided filter is a fast, non-approximate edge preserving filter. It uses
the local pixel statistics of an input image and a guide image to solve a
linear regression problem to compute the smoothed image.

The authors make the following comment: "Try the guided filter in any situation
when the bilateral filter works well. The guided filter is much faster and
sometimes (though not always) works even better." [1]_

There are two adjustable parameters: the window_size, which controls the 
size of the neighbourhood considered in computing the statistics, and eta, 
which controls the strength of the smoothing. As eta becomes very large 
compared to the image intensities the result approaches an averaging filter 
when no guide is specified.

==========
References
==========

.. [1] http://research.microsoft.com/en-us/um/people/kahe/eccv10/


"""

from skimage.data import immunohistochemistry 
from skimage.restoration import guided_filter
import numpy as np
import matplotlib.pyplot as plt


# Create a gray rectangle on a black background and add some large amplitude
# gaussian noise.
gray_square = np.zeros((400, 400))
gray_square[100:300, 100:300] = 0.5
noisy_square = gray_square + np.random.normal(scale=0.4, size=gray_square.shape)
noisy_square = np.clip(noisy_square, 0, 1)

# Guided filter the gray rectangle with itself at different values of eta.
plt.subplot(2,2,1)
plt.imshow(noisy_square, cmap='gray', vmin=0, vmax=1)
etas = [0.05, 0.2, 1]

for i, eta in enumerate(etas):
    plt.subplot(2, 2, i+2)
    guided = guided_filter(noisy_square, eta, 5)
    plt.imshow(guided, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

plt.savefig('noisy_square_self_guide.png',dpi=100)
plt.show()


plt.subplot(2,2,1)
plt.imshow(noisy_square, cmap='gray', vmin=0, vmax=1)
plt.axis('off')

# Guided filter the gray rectangle, but this time with the original sharp edged
# image as a better estimate of the image structure.
for i, eta in enumerate(etas):
    plt.subplot(2, 2, i+2)
    guided = guided_filter(noisy_square, eta, 5, guide=gray_square)
    plt.imshow(guided, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    
plt.savefig('noisy_square_sharp_guide.png',dpi=100)
plt.show()

# Load a colour image and add gaussian noise idependently to each of the
# channels of a colour image.
ihc = immunohistochemistry()/255
ihc_noisy = ihc + np.random.normal(scale=0.3, size=ihc.shape)
ihc_noisy = np.clip(ihc_noisy, 0, 1)
ihc_noisy_gray = ihc_noisy.mean(axis=2)

plt.subplot(2,2,1)
plt.imshow(ihc_noisy)
plt.axis('off')

# Guided filter the colour image, using the mean of the colour channels as a 
# guide that is less affected by the applied colour noise.
etas = [0.05,0.1,0.2]
for i, eta in enumerate(etas):
    plt.subplot(2, 2, i+2)
    guided = guided_filter(ihc_noisy, eta, 5, guide=ihc_noisy_gray)
    guided = np.clip(guided, 0, 1)    
    plt.imshow(guided)
    plt.axis('off')
    
plt.show()
plt.savefig('denoised_immunihistochemistry.png',dpi=100)
plt.show()
