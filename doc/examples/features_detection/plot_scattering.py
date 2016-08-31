"""
===================
Scattering Features
===================

This example shows how to use the scattering features for image classification.
They have shown to outperform any other 'non-learned'
image representations for image classification tasks [2]_.
Structurally, they are very similar to deep learning representations, with fixed filters, which are Wavelet filters.
Since the scattering transform is based on the Discrete Wavelet transform (DWT), its stable to deformations.
It is also invariant to small translations. For more details on its mathematical properties, see [1]_.

Here, we show how to use this implementation to access easily different scattering coefficients stored
in the scattering vectors. For more information on how to use scattering cefficients for image classification,
 please check the example 'Scattering features for supervised learning'.


 1) How to use the scattering vectors obtained from MNIST and CIFAR10 databases for classification. These are both very
 challenging databases widely used in research to compare the quality of image representations and classification methods.

 2) H

# Mathematical definition of the coefficients

Given an image $x$, the scattering transform computes (generally) three layers of cascaded convolutions
and non-linear operators:

-*Zero-order coefficients:*  $Sx[0] = x \ast \phi$

-*First-order coefficients:*   $Sx[1][(i,l)] = | x \ast \psi_{(i,l)} | \ast \phi$

-*Second-order coefficients:*  $Sx[2][(i,l)][(j,l_2)] = | | x \ast \psi_{(i,l)} | \ast \psi_{(j,l_2)}| \ast \phi$

where $\psi$ is a band-pass filter, $\phi$ is a low-pass filter (normally a Gaussian), operator $\ast$ is a
2D convolution, and $|\cdot|$ is the complex modulus. If $x$ is of size $(px,px)$, the maximum number of scales
is $J=\log_2(px)$ and $i \in [0,J-1]$. For second-order coefficients, we compute coefficients with $j>i$, since other
coefficients do not have enough energy to be significant.

# Implementation: Dictionary access to the scattering coefficients
Let's see how to access the different layers of the scattering transform. The scattering function outputs a dictionary
python structure that allows an easy access to the scattering vector (first output of the function). The keys
 for this dictionary structure are the following:

- *Zero-order*: we only have one key for the only coefficient:
                scat_tree[0]

- *First-order*: keys is a tuple with the scale and angle:
                scat_tree[(i,l)]

- *Second-order*: key is a tuple of two tuples, with the scale and angle of the first layer and then the
 scale and angle of the second layer:
                scat_tree[( (i,l)  , (j,l_2) )]

# Image Classification



..[1] Bruna, J., Mallat, S. 'Invariant Scattering Convolutional Networks'.IEEE TPAMI, 2012.
..[2] Oyallon, E. et Mallat, S. 'Deep Roto-translation Scattering for Object Classification'. CVPR 2015

"""
# Load an image

import numpy as np
import skimage.data as d
from skimage.transform import resize
import matplotlib.pylab as plt

px = 32 # size of the image (squared)
im= resize( d.camera(), (px,px))
plt.imshow(abs(im))

###### Compute the scattering

from skimage.filters.filter_bank import multiresolution_filter_bank_morlet2d
from skimage.features.scattering import scattering

J=3 #number of scales
L=8 #number of angles

m=2 # compute up to the second order scattering coeffs.

#get filters
wavelet_filters, lw = multiresolution_filter_bank_morlet2d(px, J=J, L=L)

#scattering coefficients (S) and the access tree (scat_tree)
S,u,scat_tree = scattering(im[np.newaxis,:,:], wavelet_filters,m=m)

num_images,coef_index, spatial, spatial = S.shape

zero_order_coef_index = 0
#We can find the zero-order scattering coefficients in the S vector:
S_zero_order = S[0,zero_order_coef_index,]

plt.suptitle('Zero order scattering coefficients')
plt.subplot(1,2,1)
plt.title('Using the vector structure')
plt.imshow(S_zero_order)
#or we can access them using the scat_tree structure, which is a view of the S structure.
plt.subplot(1,2,2)
plt.title('Using the dictionary structure')
plt.imshow(scat_tree[0][0,:,:])
plt.show()

# - First order coefficients
# we want to access one of the coefficients
i = 0
l = 3
S_first_order = S[0,i*L+l+1,:,:]

plt.suptitle('First order scattering coefficients')
plt.subplot(1,2,1)
plt.imshow(S_first_order)
#using the stree structure
plt.subplot(1,2,2)
plt.imshow(scat_tree[(i,l)][0,:,:])
plt.show()

# - Second order coefficients: $|x \ast \psi_i| \ast \psi_j| \ast \phi$
#The complete number of coefficients second order coefficients is: $\frac{J (J-1) L^2}{2}$
#We will just access the coefficient using the scat tree structure:
j = 1
l_2 = 5
plt.suptitle('Second order scattering coefficients')
plt.imshow(scat_tree[((i,l),(j,l_2))][0,:,:])
