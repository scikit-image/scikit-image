"""
===============================
Using geometric transformations
===============================

In this example, we will see how to use geometric transformations in the context
of image processing.
"""

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
"""

#: create using explicit parameters
tform = tf.SimilarityTransformation()
scale = 1
rotation = math.pi/2
translation = (0, 1)
tform.from_params(scale, rotation, translation)
print tform.matrix

#: create using transformation matrix
matrix = tform.matrix.copy()
matrix[1, 2] = 2
tform2 = tf.SimilarityTransformation(matrix)

"""
These transformation objects can be used to forward and reverse transform
coordinates between the source and destination coordinate systems:
"""

coord = [1, 0]
print tform2.forward(coord)
print tform2.reverse(tform.forward(coord))

"""
Image warping
=============

Geometric transformations can also be used to warp images:
"""

text = data.text()
tform.from_params(1, math.pi/4, (text.shape[0] / 2, -100))

# uses tform.reverse, alternatively use tf.warp(text, tform.reverse)
rotated = tf.warp(text, tform)
back_rotated = tf.warp(rotated, tform.forward)

plt.figure(figsize=(8, 3))
plt.subplot(131)
plt.imshow(text)
plt.axis('off')
plt.gray()
plt.subplot(132)
plt.imshow(rotated)
plt.axis('off')
plt.gray()
plt.subplot(133)
plt.imshow(back_rotated)
plt.axis('off')
plt.gray()
plt.subplots_adjust(**margins)

"""
.. image:: PLOT2RST.current_figure

Parameter estimation
====================

In addition to the basic functionality mentioned above you can also estimate the
parameters of a geometric transformation using the least-squares method.

This can amongst other things be used for image registration or rectification,
where you have a set of control points or homologous points in two images.

Let's assume we want to recognize letters on a photograph which was not taken
from the front but at a certain angle. In the simplest case of a plane paper
surface the letters are projectively distorted. Simple matching algorithms would
not be able to match such symbols. One solution to this problem would be to warp
the image so that the distortion is removed and then apply a matching algorithm:
"""

text = data.text()

src = np.array((
    (155, 15),
    (65, 40),
    (260, 130),
    (360, 95)
))
dst = np.array((
    (0, 0),
    (0, 50),
    (300, 50),
    (300, 0)
))

tform3 = tf.estimate_transformation('projective', src, dst)
warped = tf.warp(text, tform3, output_shape=(50, 300))

plt.figure(figsize=(8, 3))
plt.subplot(211)
plt.imshow(text)
plt.plot(src[:, 0], src[:, 1], '.r')
plt.axis('off')
plt.gray()
plt.subplot(212)
plt.imshow(warped)
plt.axis('off')
plt.gray()
plt.subplots_adjust(**margins)

"""
.. image:: PLOT2RST.current_figure
"""

plt.show()
