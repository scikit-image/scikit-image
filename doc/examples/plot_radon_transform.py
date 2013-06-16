"""
===============
Radon transform
===============

In computed tomography, the tomography reconstruction problem is to obtain
a tomographic slice image from a set of projections. A projection is formed
by drawing a set of parallel rays through the 2D object of interest, assigning
the integral of the object's contrast along each ray to a single pixel in the
projection. A single projection of a 2D object is one dimensional. To
enable computed tomography reconstruction of the object, several projections
must be acquired, each of them with the rays making a different angle with
the axes of the object. A collection of projections at several angles is
called a sinogram.

The inverse Radon transform is used in computed tomography to reconstruct
a 2D image from can hence be used to reconstruct an object from the measured
projections (the sinogram). A practical, exact implementation of the inverse
Radon transform does not exist, but there are several good approximate
algorithms available.

As the inverse Radon transform reconstructs the object from a set of
projections, the (forward) Radon transform can be used to simulate a
tomography experiment.

For more information see:

  - AC Kak, M Slaney, "Principles of Computerized Tomographic Imaging",
    http://www.slaney.org/pct/pct-toc.html
  - http://en.wikipedia.org/wiki/Radon_transform

This script performs the Radon transform to simulate a tomography experiment
and reconstructs the input image based on the resulting sinogram formed by
the simulation. Two methods for performing the inverse Radon transform
and reconstructing the original image will be used: The Filtered Back
Projection (FBP) and the Simultaneous Algebraic Reconstruction
Technique (SART).


The forward transform
=====================

As our original image, we will use the Shepp-Logan phantom. When calculating
the Radon transform, we need to decide how many projection angles we wish
to use. As a rule of thumb, the number of projections should be about the
same as the number of pixels there are across the object (to see why this
is so, consider how many unknown pixel values must be determined in the
reconstruction process and compare this to the number of measurements
provided by the projections), and we follow that rule here. Below is the
original image and its Radon transform, often known as its _sinogram_:
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale

image = imread(data_dir + "/phantom.png", as_grey=True)
image = rescale(image, scale=0.4)

plt.figure(figsize=(8, 4.5))

plt.subplot(121)
plt.title("Original")
plt.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(image.shape), endpoint=True)
sinogram = radon(image, theta=theta, circle=True)
plt.subplot(122)
plt.title("Radon transform\n(Sinogram)");
plt.xlabel("Projection angle (deg)");
plt.ylabel("Projection position (pixels)");
plt.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

plt.subplots_adjust(hspace=0.4, wspace=0.5)

"""
.. image:: PLOT2RST.current_figure

Reconstruction with the Filtered Back Projection (FBP)
======================================================

The mathematical foundation of the filtered back projection is the Fourier
slice theorem (http://en.wikipedia.org/wiki/Projection-slice_theorem). It
uses Fourier transform of the projection and interpolation in Fourier space
to obtain the 2D Fourier transform of the image, which is then inverted to
form the reconstructed image. The filtered back projection is among the
fastest methods of performing the inverse Radon transform. The only tunable
parameter for the FBP is the filter, which is applied to the Fourier
transformed projections. It is needed to suppress high frequency noise in the
reconstruction. ``skimage`` provides a few different options for the filter.

"""

from skimage.transform import iradon

reconstruction_fbp = iradon(sinogram, theta=theta, circle=True)

imkwargs = dict(vmin=-0.2, vmax=0.2)
plt.figure(figsize=(8, 4.5))
plt.subplot(121)
plt.title("Reconstruction\nFiltered back projection")
plt.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
plt.subplot(122)
plt.title("Reconstruction error\nFiltered back projection")
plt.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)

"""
.. image:: PLOT2RST.current_figure

Reconstruction with the Simultaneous Algebraic Reconstruction Technique
=======================================================================

Algebraic reconstruction techniques for tomography are based on a
straightforward idea: For a pixelated image the value of a single ray in a
particular projection is simply a sum of all the pixels the ray passes through
on its way through the object. This is a way of expressing the forward Radon
transform. The inverse Radon transform can then be formulated as a (large) set
of linear equations. As each ray passes through a small fraction of the pixels
in the image, this set of equations is sparse, allowing iterative solvers for
sparse linear systems to tackle the system of equations. One iterative method
has been particularly popular, namely Kaczmarz' method, which has the property
that the solution will approach a least-squares solution of the equation set.

The combination of the formulation of the reconstruction problem as a set
of linear equations and an iterative solver makes algebraic techniques
relatively flexible, hence some forms of prior knowledge can be incorporated
with relative ease.

``skimage`` provides one of the more popular variations of the algebraic
reconstruction techniques: the Simultaneous Algebraic Reconstruction Technique
(SART). It uses Kaczmarz' method as the iterative solver. A good
reconstruction is normally obtained in a single iteration, making the method
computationally effective. Running one or more extra iterations will normally
The implementation in ``skimage`` allows prior
information of the form of a lower and upper threshold on the reconstructed
values to be supplied to the reconstruction.

"""

from skimage.transform import iradon_sart

reconstruction_sart = iradon_sart(sinogram, theta=theta)

plt.figure(figsize=(8, 8.5))

plt.subplot(221)
plt.title("Reconstruction\nSART")
plt.imshow(reconstruction_sart, cmap=plt.cm.Greys_r)
plt.subplot(222)
plt.title("Reconstruction error\nSART")
plt.imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r, **imkwargs)

# Run a second iteration of SART by supplying the reconstruction
# from the first iteration as an initial estimate
reconstruction_sart2 = iradon_sart(sinogram, theta=theta,
                                   image=reconstruction_sart)

d = reconstruction_sart - image
print(d.max(), d.min())

plt.subplot(223)
plt.title("Reconstruction\nSART, 2 iterations")
plt.imshow(reconstruction_sart2, cmap=plt.cm.Greys_r)
plt.subplot(224)
plt.title("Reconstruction error\nSART, 2 iterations")
plt.imshow(reconstruction_sart2 - image, cmap=plt.cm.Greys_r, **imkwargs)

"""
.. image:: PLOT2RST.current_figure
"""
