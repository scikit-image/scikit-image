from __future__ import division

# We will visualize a 2-D vortex with different configurations of the
# ``line_integral_convolution`` algorithm, and we will use the 2-D vortex to
# add motion blur to a sample image. First the necessary modules and functions
# need to be imported.

import numpy as np
from skimage.filters import line_integral_convolution
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from skimage.data import camera

if rcParams['savefig.dpi'] == 'figure':
    dpi = rcParams['figure.dpi']
else:
    dpi = rcParams['savefig.dpi']

# The 2-D vortex that we want to visualize is described by the ``velocity``
# array.

position = np.mgrid[:150, :180].astype(float)
velocity = np.tensordot((position -
                         np.array(position.shape[1:])[:, None, None] / 2),
                        [[0, 1], [-1, 0]], axes=(0, 0))
# squared length of the velocity
velocity_ssq = np.sum(np.square(velocity), axis=-1)

# The following lines create three random boolean images. ``image5`` contains
# an equal amount of black an white pixels. ``image01`` is mostly black and
# contains only a few white pixels. ``image004`` is darker at pixels with
# higher velocity, with a minimum brightness defined by the ``0.004``
# constant and a maximum brightness controlled by the ``0.2`` constant.

np.random.seed(0)
image5 = (np.random.random(position.shape[1:]) < 0.5)
np.random.seed(0)
image01 = (np.random.random(position.shape[1:]) < 0.01)
np.random.seed(0)
image004 = (np.maximum(np.sqrt(velocity_ssq / np.max(velocity_ssq)),
                       0.2) *
            np.random.random(position.shape[1:])) < 0.004

# A symmetric Gaussian kernel (``gauss_kernel``) and an asymmetric exponential
# kernel (``exp_kernel``) are created.

gauss_kernel = norm.pdf(np.linspace(-3, 3, 25))
exp_kernel = np.exp(-np.linspace(0, 3, 25))

# The symmetric kernel in combination with the ``image5`` input
# image can be used to visualize the flow, ignoring velocity magnitude and
# direction.

fig = plt.figure(figsize=(np.array(image5.T.shape) / dpi))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(np.clip(line_integral_convolution(image5, velocity,
                                            gauss_kernel), 0, 1),
          cmap='Greys_r', interpolation='nearest')
plt.figtext(0.5, 0.05, 'Standard LIC', fontsize=1000 // dpi,
            color='w', backgroundcolor='k', ha='center')

# The asymmetric kernel can be used to visualize the flow direction. The
# ``image01`` array is used instead of the ``image5`` image, and the origin is
# set to ``None``. Also, ``weighted`` is set to ``'integral'`` to treat the
# white pixels in the input image as single particles.

fig = plt.figure(figsize=(np.array(image01.T.shape) / dpi))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(np.clip(line_integral_convolution(image01, velocity,
                                            exp_kernel,
                                            origin=None,
                                            weighted='integral'),
                  0, 1),
          cmap='Greys_r', interpolation='nearest')
plt.figtext(0.5, 0.05, 'Direction', fontsize=1000 // dpi,
            color='w', backgroundcolor='k', ha='center')

# The symmetric kernel in combination with ``step_size='unit_time'`` and
# ``maximum_velocity=2.`` visualizes the velocity magnitude and ignores the
# direction. The largest element in the kernel is set to 1 in this case
# (using ``gauss_kernel / np.max(gauss_kernel)``). To ensure that the resulting
# image has the same line density everywhere, we adjust the number of white
# pixels to the velocity by using ``image004``.

fig = plt.figure(figsize=(np.array(image004.T.shape) / dpi))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(np.clip(line_integral_convolution(image004, velocity,
                                            (gauss_kernel /
                                             np.max(gauss_kernel)),
                                            weighted='integral',
                                            step_size='unit_time',
                                            maximum_velocity=2.),
                  0, 1),
          cmap='Greys_r', interpolation='nearest')
plt.figtext(0.5, 0.05, 'Magnitude', fontsize=1000 // dpi,
            color='w', backgroundcolor='k', ha='center')

# Velocity direction and magnitude can be visualized by using the same
# configuration with the asymmetric kernel.

fig = plt.figure(figsize=(np.array(image004.T.shape) / dpi))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(np.clip(line_integral_convolution(image004, velocity,
                                            exp_kernel,
                                            origin=None,
                                            weighted='integral',
                                            step_size='unit_time',
                                            maximum_velocity=2.),
                  0, 1),
          cmap='Greys_r', interpolation='nearest')
plt.figtext(0.5, 0.05, 'Direction and magnitude', fontsize=1000 // dpi,
            color='w', backgroundcolor='k', ha='center')

# If we replace ``weighted='integral'`` with ``weighted='average'``
# (the default), we can add motion blur to an image.

imageC = camera()
try:
    from scipy.misc import imresize
    imageC = imresize(imageC, 150 / np.max(imageC.shape))
    position = np.mgrid[[slice(None, s)
                         for s in imageC.shape]].astype(float)
    velocity = np.tensordot((position -
                             np.array(imageC.shape)[:, None, None] / 2),
                            [[0, 1], [-1, 0]], axes=(0, 0))
    fig = plt.figure(figsize=(np.array(imageC.T.shape) / dpi))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.clip(line_integral_convolution(imageC, velocity,
                                                exp_kernel,
                                                origin=None,
                                                step_size='unit_time',
                                                maximum_velocity=2.),
                      0, 255),
              cmap='Greys_r', interpolation='nearest')
    plt.figtext(0.5, 0.05, 'Motion blur', fontsize=1000 // dpi,
                color='w', backgroundcolor='k', ha='center')
except ImportError as e:
    import sys
    sys.stderr.write('WARNING: skipping motion blur test: %s\n' % e)
plt.show()
