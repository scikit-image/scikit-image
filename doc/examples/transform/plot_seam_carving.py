"""
============
Seam Carving
============

This example demonstrates how images can be resized using seam carving [1]_.
Resizing to a new aspect ratio distorts image contents. Seam carving attempts
to resize *without* distortion, by removing regions of an image which are less
important. In these examples we are using the Sobel filter and the forward
energy filter [2]_ to signify the importance of each pixel.

.. [1] Shai Avidan and Ariel Shamir
       "Seam Carving for Content-Aware Image Resizing"
       http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Avidan07.pdf

.. [2] Michael Rubinstein, Ariel Shamir, and Shai Avidan
        "Improved Seam Carving for Video Retargeting"
        http://www.faculty.idc.ac.il/arik/SCWeb/vidret/index.html

"""
from skimage import data, draw
from skimage import transform, util
import numpy as np
from skimage import filters, color
from matplotlib import pyplot as plt

hl_color = np.array([0, 1, 0])

img = data.rocket()
img = util.img_as_float(img)
eimg = filters.sobel(color.rgb2gray(img))

plt.title('Original Image')
plt.imshow(img)

######################################################################

resized = transform.resize(img, (img.shape[0], img.shape[1] - 200),
                           mode='reflect')
plt.figure()
plt.title('Resized Image')
plt.imshow(resized)

######################################################################
# Using the Sobel filter, the importance of each pixel is determined
# by using edge detection, where brighter pixels are more important.

plt.figure()
plt.title('Original Energy Image')
plt.imshow(eimg)

######################################################################
# And we can remove unimportant pixels following this seam carving
# pattern:


def seam_carve(img, f, mode, n, freq=1):
    """Helper function to recalculate the energy map after seam removal

    Parameters
    ----------
    img : (M, N) or (M, N, 3) ndarray
        Input image whose seams are to be removed.
    f : lambda img, mode : (M, N) ndarray
        The energy map function to calculate importance of each pixel
        in `img` in the `mode` direction
    mode : str {'horizontal', 'vertical'}
    n : int
        Number of seams to be removed.
    freq : int, optional
        How many seams to remove before recalculating energy map

    Returns
    -------
    out : ndarray
        The cropped image with the seams removed.
    """
    for i in range(0, n, freq):
        eimg = f(img, mode)
        num = freq if i + freq < n else n - i
        img = transform.seam_carve(img, eimg, mode, num)
    return img


def energy(img, mode):
    """
    Parameters
    ----------
    img : (M, N) or (M, N, 3) ndarray
        Input image whose seams are to be removed.
    mode : str {'horizontal', 'vertical'}
        Ignored in this case.
    """
    return filters.sobel(color.rgb2gray(img))


out = seam_carve(img, energy, 'vertical', 200)
plt.figure()
plt.title('Resized using Seam Carving')
plt.imshow(out)

######################################################################
# Resizing distorts the rocket and surrounding objects, whereas seam carving
# removes empty spaces and preserves object proportions.
#
# Object Removal
# --------------
#
# Seam carving can also be used to remove artifacts from images. This
# requires weighting the artifact with low values. Recall lower weights are
# preferentially removed in seam carving. The following code masks the
# rocket's region with low weights, indicating it should be removed.
# For simplicity, we do not update the mask and energy map after each
# seam removal.

masked_img = img.copy()

poly = [(404, 281), (404, 360), (359, 364), (338, 337), (145, 337), (120, 322),
        (145, 304), (340, 306), (362, 284)]
pr = np.array([p[0] for p in poly])
pc = np.array([p[1] for p in poly])
rr, cc = draw.polygon(pr, pc)

masked_img[rr, cc, :] = masked_img[rr, cc, :]*0.5 + hl_color*.5
plt.figure()
plt.title('Object Marked')

plt.imshow(masked_img)

######################################################################

eimg[rr, cc] -= 1000

plt.figure()
plt.title('Object Removed')
out = transform.seam_carve(img, eimg, 'vertical', 90)
resized = transform.resize(img, out.shape, mode='reflect')
plt.imshow(out)
plt.show()

######################################################################
# Forward Energy
# --------------
#
# While calculating the energy of a pixel using edge detection
# worked well for the rocket image, it gives non ideal results
# with certain other images like this bench:

img = data.bench()
plt.figure()
plt.title('Original Image')
plt.imshow(img)

######################################################################

plt.figure()
plt.title('Original Energy Image')
plt.imshow(energy(img, 'vertical'))

######################################################################

out = seam_carve(img, energy, 'vertical', 400)
plt.figure()
plt.title('Resized using Seam Carving')
plt.imshow(out)

######################################################################
# Seam carving using the Sobel edge filter distorts the person
# and unncessarily preserves pixels along the bench and the fence
#
# This is why Rubinstein et al proposed the forward energy filter,
# which calculates the importance of a pixel based on whether
# removing that pixel would create new edges:


def forward_energy(img, mode):
    """
    Parameters
    ----------
    img : (M, N) or (M, N, 3) ndarray
        Input image whose seams are to be removed.
    mode : str {'horizontal', 'vertical'}
    """
    return filters.forward_energy(color.rgb2gray(img), mode)


plt.figure()
plt.title('Original Forward Energy Image')
plt.imshow(forward_energy(img, 'vertical'))

######################################################################

out = seam_carve(img, forward_energy, 'vertical', 400)
plt.figure()
plt.title('Resized using Forward Energy Seam Carving')
plt.imshow(out)

######################################################################
# This results in much better output.

######################################################################
