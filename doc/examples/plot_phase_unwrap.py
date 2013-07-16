"""
================
Phase Unwrapping
================

Some signals can only be observed modulo 2*pi, and this can also apply to
two- and three dimensional images. In these cases phase unwrapping is
needed to recover the underlying, unwrapped signal. In this example we will
demonstrate an algorithm [1]_ implemented in ``skimage`` at work for such a
problem. One-, two- and three dimensional images can all be unwrapped using
skimage. Here we will demonstrate phase unwrapping in the two dimensional case.
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
plt.imshow(image_unwrapped, cmap='gray')
plt.colorbar()

plt.subplot(224)
plt.title('Unwrapped minus original')
plt.imshow(image_unwrapped - image, cmap='gray')
plt.colorbar()

"""
.. image:: PLOT2RST.current_figure

The unwrapping procedure accepts masked arrays, and can also optionally
assume cyclic boundaries to connect edges of an image. In the example below,
we study a simple phase ramp which has been split in two by masking
a row of the image.
"""

# Create a simple ramp
image = np.ones((100, 100)) * np.linspace(0, 8 * np.pi, 100).reshape((-1, 1))
# Mask the image to split it in two horizontally
mask = np.zeros_like(image, dtype=np.bool)
mask[image.shape[0] // 2, :] = True

image_wrapped = np.ma.array(np.angle(np.exp(1j * image)), mask=mask)
# Unwrap image without wrap around
image_unwrapped_no_wrap_around = unwrap_phase(image_wrapped,
                                              wrap_around=(False, False))
# Unwrap with wrap around enabled for the 0th dimension
image_unwrapped_wrap_around = unwrap_phase(image_wrapped,
                                           wrap_around=(True, False))

plt.figure()
plt.subplot(221)
plt.title('Original')
plt.imshow(np.ma.array(image, mask=mask), cmap='jet')
plt.colorbar()

plt.subplot(222)
plt.title('Wrapped phase')
plt.imshow(image_wrapped, cmap='jet', vmin=-np.pi, vmax=np.pi)
plt.colorbar()

plt.subplot(223)
plt.title('Unwrapped without wrap_around')
plt.imshow(image_unwrapped_no_wrap_around, cmap='jet')
plt.colorbar()

plt.subplot(224)
plt.title('Unwrapped with wrap_around')
plt.imshow(image_unwrapped_wrap_around, cmap='jet')
plt.colorbar()

plt.show()

"""
.. image:: PLOT2RST.current_figure

In the figures above, the masked row can be seen as a white line across
the image. The difference between the two unwrapped images in the bottom row
is clear: Without unwrapping (lower left), the regions above and below the
masked boundary do not interact at all, resulting in an offset between the
two regions of an arbitrary integer times two pi. We could just as well have
unwrapped the regions as two separate images. With wrap around enabled for the
vertical direction (lower rigth), the situation changes: Unwrapping paths are
now allowed to pass from the bottom to the top of the image and vice versa, in
effect providing a way to determine the offset between the two regions.

References
----------

.. [1] Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor,
       and Munther A. Gdeisat, "Fast two-dimensional phase-unwrapping
       algorithm based on sorting by reliability following a noncontinuous
       path", Journal Applied Optics, Vol. 41, No. 35, pp. 7437, 2002
"""
