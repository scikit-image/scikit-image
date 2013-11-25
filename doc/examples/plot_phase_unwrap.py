"""
================
Phase Unwrapping
================

Some signals can only be observed modulo 2*pi, and this can also apply to
two- and three dimensional images. In these cases phase unwrapping is
needed to recover the underlying, unwrapped signal. In this example we will
demonstrate an algorithm [1]_ implemented in ``skimage`` at work for such a
problem.
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage import data, img_as_float, color, exposure
from skimage.exposure import unwrap_phase


# Load an image as a floating-point grayscale
image = color.rgb2gray(img_as_float(data.chelsea()))
# Scale the image to [0, 4*pi]
image = exposure.rescale_intensity(image, out_range=(0, 4 * np.pi))
# Create a phase-wrapped image in the interval [-pi, pi)
image_wrapped = np.angle(np.exp(1j * image))
# Perform phase unwrapping
image_unwrapped = unwrap_phase(image_wrapped)

plt.figure()
plt.subplot(221)
plt.title('Original')
plt.imshow(image, cmap='gray', vmin=0, vmax=4 * np.pi)
plt.colorbar()

plt.subplot(222)
plt.title('Wrapped phase')
plt.imshow(image_wrapped, cmap='gray', vmin=-np.pi, vmax=np.pi)
plt.colorbar()

plt.subplot(223)
plt.title('After phase unwrapping')
plt.imshow(image_unwrapped,  cmap='gray')
plt.colorbar()

plt.subplot(224)
plt.title('Unwrapped minus original')
plt.imshow(image_unwrapped - image,  cmap='gray')
plt.colorbar()

plt.show()


"""
.. image:: PLOT2RST.current_figure

References
----------

.. [1] R. M. Goldstein, H. A. Zebker, C. L. Werner, "Satellite radar
       interferometry: Two-dimensional phase unwrapping", Radio Science 23
       (1988) 4, pp 713--720.
"""
