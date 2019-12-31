"""
===================================
Types of geometric transformations
===================================

The different types of geometry transformations available in scikit-image are
shown here, by increasing order of complexity. While we focus here on the
mathematical properties of transformations, tutorial :ref:`sphx_glr_auto_examples_transform_plot_geometric.py`
explains how to use such transformations for various tasks such as image warping or parameter estimation.
"""
import math
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import transform as tf
from skimage import img_as_float

######################################################################
# Euclidean (rigid) transformations
# =================================
#
# A `Euclidean transformation <https://en.wikipedia.org/wiki/Rigid_transformation>`_,
# also called rigid transformation, preserves the Euclidean distance between pairs of
# points. It can be described as a rotation followed by a translation.

tform = tf.EuclideanTransform(
        rotation = np.pi / 12.,
        #translation = (100, -20)
        )
print(tform.params)

######################################################################
# Now let's apply this transformation to an image. We use the inverse
# of ``tform`` since ``tform`` is a transformation of *coordinates*,
# therefore we need to use its inverse to rotate and shift the image 
# with the given parameters.
#
# Note that the first dimension corresponding to the horizontal axis,
# positive from left to right, while the second dimension is the vertical
# axis, positive downwards.

img = img_as_float(data.camera())
tf_img = tf.warp(img, tform.inverse)
_ = plt.imshow(tf_img)

######################################################################
# The rotation is around the origin, that is the top-left corner of the
# image. For a rotation, it is possible to compose 

angle = np.pi / 3
center = np.array([img.shape[0] / 2, img.shape[1] / 2])
shift = [center[1] * (1 - math.cos(angle)) + center[0] * math.sin(angle),
         center[0] * (1 - math.cos(angle)) - center[1] * math.sin(angle)]

tform1 = tf.EuclideanTransform(rotation=angle)
tform2 = tf.EuclideanTransform(translation=-center)
matrix = np.linalg.inv(tform2.params) @ tform1.params @ tform2.params

tform_composed = tf.EuclideanTransform(matrix)

tform_all = tf.EuclideanTransform(rotation=angle, translation=shift)
tf_img = tf.warp(img, tform_composed.inverse)
plt.figure()
_ = plt.imshow(tf_img)

plt.show()
