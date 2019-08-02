"""
==============
Edge operators
==============

Edge operators are used in image processing within edge detection algorithms.
They are discrete differentiation operators, computing an approximation of the
gradient of the image intensity function.

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.filters import roberts, sobel, sobel_h, sobel_v, scharr, \
    scharr_h, scharr_v, prewitt, prewitt_v, prewitt_h, farid_v, farid_h

image = camera()
edge_roberts = roberts(image)
edge_sobel = sobel(image)

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                       figsize=(8, 4))

ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
ax[0].set_title('Roberts Edge Detection')

ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
ax[1].set_title('Sobel Edge Detection')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

######################################################################
# Different operators compute different finite-difference approximations of
# the gradient. For example, the Scharr filter results in a less rotational
# variance than the Sobel filter that is in turn better than the Prewitt
# filter [1]_ [2]_ [3]_. The difference between the Prewitt and Sobel filters
# and the Scharr filter is illustrated below with an image that is the
# discretization of a rotation- invariant continuous function. The
# discrepancy between the Prewitt and Sobel filters, and the Scharr filter is
# stronger for regions of the image where the direction of the gradient is
# close to diagonal, and for regions with high spatial frequencies. For the
# example image the differences between the filter results are very small and
# the filter results are visually almost indistinguishable.
#
# .. [1] https://en.wikipedia.org/wiki/Sobel_operator#Alternative_operators
#
# .. [2] B. Jaehne, H. Scharr, and S. Koerkel. Principles of filter design.
#        In Handbook of Computer Vision and Applications. Academic Press,
#        1999.
#
# .. [3] https://en.wikipedia.org/wiki/Prewitt_operator

x, y = np.ogrid[:100, :100]
# Rotation-invariant image with different spatial frequencies
img = np.exp(1j * np.hypot(x, y) ** 1.3 / 20.).real

edge_sobel = sobel(img)
edge_scharr = scharr(img)
edge_prewitt = prewitt(img)

diff_scharr_prewitt = edge_scharr - edge_prewitt
diff_scharr_sobel = edge_scharr - edge_sobel
max_diff = np.max(np.maximum(diff_scharr_prewitt, diff_scharr_sobel))

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                         figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(edge_scharr, cmap=plt.cm.gray)
ax[1].set_title('Scharr Edge Detection')

ax[2].imshow(diff_scharr_prewitt, cmap=plt.cm.gray, vmax=max_diff)
ax[2].set_title('Scharr - Prewitt')

ax[3].imshow(diff_scharr_sobel, cmap=plt.cm.gray, vmax=max_diff)
ax[3].set_title('Scharr - Sobel')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

######################################################################
# As in the previous example, here we illustrate the rotational invariance of
# the filters. The top row shows a rotationally invariant image along with the
# angle of its analytical gradient. The other two rows contain the difference
# between the different gradient approximations (Sobel, Prewitt, Scharr &
# Farid) and analytical gradient.
#
# The Farid & Simoncelli derivative filters [4]_, [5]_  are the most
# rotationally invariant, but require a 5x5 kernel, which is computationally
# more intensive than a 3x3 kernel.
#
# .. [4] Farid, H. and Simoncelli, E. P., "Differentiation of discrete
#        multidimensional signals", IEEE Transactions on Image Processing 13(4):
#        496-508, 2004. :DOI:`10.1109/TIP.2004.823819`
#
# .. [5] Wikipedia, "Farid and Simoncelli Derivatives." Available at:
#        <https://en.wikipedia.org/wiki/Image_derivatives#Farid_and_Simoncelli_Derivatives>


x, y = np.mgrid[-10:10:255j, -10:10:255j]
img = np.sin(x ** 2 + y ** 2)

imgx = 2 * x * np.cos(x ** 2 + y ** 2)
imgy = 2 * y * np.cos(x ** 2 + y ** 2)


def angle(dx, dy):
    return np.mod(np.arctan2(dy, dx), np.pi)


true_angle = angle(imgx, imgy)

angle_farid = angle(farid_h(img), farid_v(img))
angle_sobel = angle(sobel_h(img), sobel_v(img))
angle_scharr = angle(scharr_h(img), scharr_v(img))
angle_prewitt = angle(prewitt_h(img), prewitt_v(img))


def diff_angle(angle_1, angle_2):
    return np.minimum(np.pi - np.abs(angle_1 - angle_2),
                      np.abs(angle_1 - angle_2))


diff_farid = diff_angle(true_angle, angle_farid)
diff_sobel = diff_angle(true_angle, angle_sobel)
diff_scharr = diff_angle(true_angle, angle_scharr)
diff_prewitt = diff_angle(true_angle, angle_prewitt)

fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True,
                         figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(true_angle, cmap=plt.cm.hsv)
ax[1].set_title('Analytical gradient angle')

ax[2].imshow(diff_sobel, cmap=plt.cm.inferno, vmin=0, vmax=0.02)
ax[2].set_title('Sobel error')

ax[3].imshow(diff_prewitt, cmap=plt.cm.inferno, vmin=0, vmax=0.02)
ax[3].set_title('Prewitt error')

ax[4].imshow(diff_scharr, cmap=plt.cm.inferno, vmin=0, vmax=0.02)
ax[4].set_title('Scharr error')

cax = ax[5].imshow(diff_farid, cmap=plt.cm.inferno, vmin=0, vmax=0.02)
ax[5].set_title('Farid error')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.90, 0.10, 0.02, 0.50])
fig.colorbar(cax, cax=cbar_ax, ticks=[0, 0.01, 0.02])

for a in ax:
    a.axis('off')

plt.show()
