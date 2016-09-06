"""
===================
Scattering Features
===================

This example shows how to use the scattering transform and how to access easily different scattering coefficients stored
in the output vector. For more information on how to use scattering coefficients for image classification,
please check the example *Scattering features for supervised learning*.

We call the scattering features (or coefficients) the output of the scattering transform applied to an image. These features
have shown to outperform any other 'non-learned' image representations for image classification tasks [2]_.
Structurally, they are very similar to deep learning representations, with fixed filters, which are Morlet filters, in this
implementation.
Since the scattering transform is based on the Discrete Wavelet transform (DWT), its stable to deformations.
It is also invariant to small translations. For more details on its mathematical properties, see [1]_.

**Definition of the scattering coefficients**

Given an image $x$, the scattering transform computes (generally) three layers of cascaded convolutions
and non-linear operators:

-*Zero-order coefficients:*  $Sx[0] = x \ast \phi$

-*First-order coefficients:*   $Sx[1][(i,l)] = | x \ast \psi_{(i,l)} | \ast \phi$

-*Second-order coefficients:*  $Sx[2][(i,l)][(j,l_2)] = | | x \ast \psi_{(i,l)} | \ast \psi_{(j,l_2)}| \ast \phi$

where $\psi$ is a band-pass filter, $\phi$ is a low-pass filter (normally a Gaussian), operator $\ast$ is a
2D convolution, and $|\cdot|$ is the complex modulus. If $x$ is of size $(px,px)$, the maximum number of scales
is $J=\log_2(px)$ and $i \in [0,J-1]$. For second-order coefficients, we compute coefficients with $j>i$, since other
coefficients do not have enough energy to be significant.

**Implementation: Dictionary access to the scattering coefficients**

Let's see how to access the different layers of the scattering transform. The scattering function outputs a python
dictionary structure that allows an easy access to the scattering coefficients (first output of the function). The keys
for this dictionary structure are the following:

-*Zero-order*: we only have one key for the only coefficient:
                            scat_tree[0]

-*First-order*: keys is a tuple with the scale $i$ and angle $l$:
                            scat_tree[(i,l)]

-*Second-order*: key is a tuple of two tuples, with the scale $i$ and angle $l$ of the first layer and then the scale $j$ and angle $l_2$ of the second layer:
                            scat_tree[( (i,l)  , (j,l_2) )]


Note that the output of any layer is a matrix of size (Num_images, spatial_dimensions, spatial_dimensions).

.. [1] Bruna, J., Mallat, S. 'Invariant Scattering Convolutional Networks'.IEEE TPAMI, 2012.
.. [2] Oyallon, E. et Mallat, S. 'Deep Roto-translation Scattering for Object Classification'. CVPR 2015

"""


import numpy as np
import skimage.data as d
from skimage.transform import resize
import matplotlib.pylab as plt
from skimage.filters.filter_bank import multiresolution_filter_bank_morlet2d
from skimage.feature.scattering import scattering

# Load an image
px = 32 # size of the image (squared)
im= resize( d.camera(), (px,px))
plt.imshow(abs(im))

# Compute the scattering
J=3 #number of scales
L=8 #number of angles
m=2 # compute up to the second order scattering coeffs.

#get filters
wavelet_filters, lw = multiresolution_filter_bank_morlet2d(px, J=J, L=L)

#scattering coefficients (S) and the access tree (scat_tree)
S,u,scat_tree = scattering(im[np.newaxis,:,:], wavelet_filters,m=m)
num_images,coef_index, spatial, spatial = S.shape

###############################
# Zero order scattering coefficients
zero_order_coef_index = 0
#We can find the zero-order scattering coefficients in the S vector:
S_zero_order = S[0,zero_order_coef_index,]

plt.figure(figsize=(16,8))
plt.suptitle('Zero order scattering coefficients')
plt.subplot(1,2,1)
plt.title('Using the vector structure')
plt.imshow(S_zero_order)
# or we can access them using the scat_tree structure, which is a view of the S structure.
plt.subplot(1,2,2)
plt.title('Using the dictionary structure')
plt.imshow(scat_tree[0][0, :, :])
plt.show()


###############################
# First order scattering coefficients

# we want to access one of the coefficients
i = 0
l = 3
S_first_order = S[0, i*L+l+1, :, :]

plt.figure(figsize=(16,8))
plt.suptitle('First order scattering coeff. (i,l)=(' + str(i) + ',' + str(l) + ')')
plt.subplot(1, 2, 1)
plt.imshow(S_first_order)
# using the stree structure
plt.subplot(1,2,2)
plt.imshow(scat_tree[(i,l)][0,:,:])
plt.show()

#########################
# Second order coefficients:
# $|x \ast \psi_i| \ast \psi_j| \ast \phi$
# The complete number of coefficients second order coefficients is: $\frac{J (J-1) L^2}{2}$
# We will just access the coefficient using the scat tree structure:
j = 1
l_2 = 5
plt.figure(figsize=(16,8))
plt.suptitle('Second order scattering coefficients')
plt.title('(i,l)=(' + str(i) + ',' + str(l) + '),(j,l_2)=(' + str(i) + ',' + str(l) + ')')
plt.imshow(scat_tree[((i, l), (j, l_2))][0, :, :])
