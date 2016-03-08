"""
===============================
Using geometric transformations
===============================

In this example, we will see how to use geometric transformations in the context
of image processing.
"""

from __future__ import print_function

import math
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import transform as tf

margins = dict(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)

"""
Basics
======

Several different geometric transformation types are supported: similarity,
affine, projective and polynomial.

Geometric transformations can either be created using the explicit parameters
(e.g. scale, shear, rotation and translation) or the transformation matrix:

First we create a transformation using explicit parameters:
"""

tform = tf.SimilarityTransform(scale=1, rotation=math.pi / 2,
                               translation=(0, 1))
print(tform.params)

"""
Alternatively you can define a transformation by the transformation matrix
itself:
"""

matrix = tform.params.copy()
matrix[1, 2] = 2
tform2 = tf.SimilarityTransform(matrix)

"""
These transformation objects can then be used to apply forward and inverse
coordinate transformations between the source and destination coordinate
systems:
"""

coord = [1, 0]
print(tform2(coord))
print(tform2.inverse(tform(coord)))

"""
Image warping
=============

Geometric transformations can also be used to warp images:
"""

text = data.text()

tform = tf.SimilarityTransform(scale=1, rotation=math.pi / 4,
                               translation=(text.shape[0] / 2, -100))

rotated = tf.warp(text, tform)
back_rotated = tf.warp(rotated, tform.inverse)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 3))
fig.subplots_adjust(**margins)
plt.gray()
ax1.imshow(text)
ax1.axis('off')
ax2.imshow(rotated)
ax2.axis('off')
ax3.imshow(back_rotated)
ax3.axis('off')

"""
.. image:: PLOT2RST.current_figure

Parameter estimation
====================

In addition to the basic functionality mentioned above you can also estimate the
parameters of a geometric transformation using the least-squares method.

This can amongst other things be used for image registration or rectification,
where you have a set of control points or homologous/corresponding points in two
images.

Let's assume we want to recognize letters on a photograph which was not taken
from the front but at a certain angle. In the simplest case of a plane paper
surface the letters are projectively distorted. Simple matching algorithms would
not be able to match such symbols. One solution to this problem would be to warp
the image so that the distortion is removed and then apply a matching algorithm:
"""

text = data.text()

src = np.array((
    (0, 0),
    (0, 50),
    (300, 50),
    (300, 0)
))
dst = np.array((
    (155, 15),
    (65, 40),
    (260, 130),
    (360, 95)
))

tform3 = tf.ProjectiveTransform()
tform3.estimate(src, dst)
warped = tf.warp(text, tform3, output_shape=(50, 300))

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 3))
fig.subplots_adjust(**margins)
plt.gray()
ax1.imshow(text)
ax1.plot(dst[:, 0], dst[:, 1], '.r')
ax1.axis('off')
ax2.imshow(warped)
ax2.axis('off')

"""
.. image:: PLOT2RST.current_figure
"""

plt.show()
