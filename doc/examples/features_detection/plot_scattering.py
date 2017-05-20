"""
===================
Scattering Features
===================

This example shows how to use the scattering transform on images.
It explains how to access easily different scattering
coefficients stored in the output vector. For more information on
how to use scattering coefficients for image
classification, please check the example *Scattering features
for supervised learning*.
We call the scattering features (or coefficients) the output of the
scattering transform applied to an image.
These features have been shown to outperform any other 'non-learned'
image representations for image
classification tasks [2]_. Structurally, they are very similar to
deep learning representations, with fixed filters
(here Morlet filters).
Since the scattering transform is based on the
Discrete Wavelet transform (DWT), it is stable to deformations.
It is also invariant to small translations. For more details on
its mathematical properties, see [1]_.

**Definition of the scattering coefficients**

Given an image :math:`x`, the scattering transform computes
(generally) three layers of cascaded convolutions
and non-linear operators:

-*Zero-order coefficients:*
:math:`Sx[0] = x * \phi`

-*First-order coefficients:*
:math:`Sx[1][(i,l)] = | x * \psi_{(i,l)} | * \phi`

-*Second-order coefficients:*
:math:`Sx[2][(i,l)][(j,l_2)]=|| x * \psi_{(i,l)}|*\psi_{(j,l_2)}|*\phi`,
 where :math:`\psi` is a band-pass filter, :math:`\phi` is a low-pass filter
 (normally a Gaussian), operator :math:`*` is a 2D convolution,
 and :math:`|\cdot|` is the complex modulus. If :math:`x` is of size
 :math:`(px,px)`, the maximum number of scales
 is :math:`J=\log_2(px)` and :math:`i \in [0,J-1]`. For every scale
 :math:`i`, we have :math:`L` angles, thus
 :math:`l \in (0,L)`. For second-order coefficients, we compute coefficients
 with :math:`j>i`, since other
 coefficients do not have enough energy to be significant.

Note that we have one zero-order, :math:`JL` first order,
and :math:`J L^2 (J-1) /2` second order coefficients.

**Implementation: Dictionary access to the scattering coefficients**

Let's see how to access the different layers of the scattering transform.
The scattering function outputs a python
dictionary structure that allows an easy access to the scattering
coefficients (first output of the function). The
*keys* for this dictionary structure are the following:

-Zero-order coefficients: scat_tree[0]
 we only have one key for the only coefficient.

-First-order coefficients: scat_tree[(i,l)]
 the keys are tuples with the scale :math:`i` and angle :math:`l`.

-Second-order coefficients: scat_tree[( (i,l)  , (j,l_2) )]
 the key is a tuple of two tuples, with the
 scale :math:`i` and angle :math:`l` of the first order and
 scale :math:`j` and angle :math:`l_2` of the second order.

Note that the output of any layer is a matrix of size
(Num_images, spatial_dimensions, spatial_dimensions).

.. [1] Bruna, J., Mallat, S. 'Invariant Scattering Convolutional Networks'.
 IEEE TPAMI, 2012.

.. [2] Oyallon, E. et Mallat, S. 'Deep Roto-translation Scattering
 for Object Classification'. CVPR 2015
"""
import numpy as np
import skimage.data as d
from skimage.transform import resize
import matplotlib.pylab as plt
from skimage.filters.filter_bank import multiresolution_filter_bank_morlet2d
from skimage.feature.scattering import scattering
# Load an image
px = 32  # size of the image (squared)
im = resize(d.camera(), (px, px))
plt.figure(figsize=(6, 3))
plt.imshow(abs(im))
plt.show()
# Compute the scattering
J = 3  # number of scales
L = 8  # number of angles
m = 2  # compute up to the second order scattering coeffs.
# get filters
wavelet_filters, lw = multiresolution_filter_bank_morlet2d(px, J=J, L=L)
# scattering coefficients (S) and the access tree (scat_tree)
S, u, scat_tree = scattering(im[np.newaxis, :, :], wavelet_filters, m=m)
num_images, coef_index, spatial, spatial = S.shape
###############################
# Zero order scattering coefficients
zero_order_coef_index = 0
# We can find the zero-order scattering coefficients in the S vector:
S_zero_order = S[0, zero_order_coef_index, ]
plt.figure(figsize=(8, 4))
plt.suptitle('Zero order scattering coefficients')
plt.subplot(1, 2, 1)
plt.title('Using the vector structure')
plt.imshow(S_zero_order)
# or we can access them using the scat_tree structure,
#  which is a view of the S structure.
plt.subplot(1, 2, 2)
plt.title('Using the dictionary structure')
plt.imshow(scat_tree[0][0, :, :])
plt.show()
###############################
# First order scattering coefficients
# we want to access one of the coefficients
i = 0
l = 3
S_first_order = S[0, i*L+l+1, :, :]
plt.figure(figsize=(8, 3))
plt.suptitle('First order scattering coeff. (i,l)=(' +
             str(i) + ',' + str(l) + ')')
plt.subplot(1, 2, 1)
plt.imshow(S_first_order)
# using the stree structure
plt.subplot(1, 2, 2)
plt.imshow(scat_tree[(i, l)][0, :, :])
plt.show()
#########################
# Second order coefficients:
# :math:`||x * \psi_{(i,l)}| * \psi_{(j,l_2)}| * \phi`
# The complete number of coefficients second order
# coefficients is: :math:`\frac{J (J-1) L^2}{2}`
# We will just access the coefficient using the scat tree structure:
j = 1
l_2 = 5
plt.figure(figsize=(8, 4))
plt.suptitle('Second order scattering coefficients: (i,l)=(' +
             str(i) + ',' + str(l) + '),(j,l_2)=(' +
             str(i) + ',' + str(l) + ')')
plt.imshow(scat_tree[((i, l), (j, l_2))][0, :, :])
