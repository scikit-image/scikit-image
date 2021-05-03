"""
======================
Simple image stitching
======================

This example demonstrates how a set of images can be assembled under
the hypothesis of rigid body motions.

"""

from matplotlib import pyplot as plt
import numpy as np
from skimage.data import moon
from skimage.util import random_noise, img_as_float
from skimage.transform import warp, AffineTransform, rotate
from skimage.feature import corner_harris, corner_peaks
from skimage.measure import ransac
from skimage.filters import gaussian
from skimage.metrics import peak_signal_noise_ratio


def match_locations(img0, img1, coords0, coords1, radius=7, sigma=3):
    """Image locations matching using SSD minimization.

    Areas from `img0` are matched with areas from `img1`. These areas
    are defined as patches located around pixels with gaussian
    weights.

    Parameters:
    -----------
    img0, img1 : 2D array
        Input images.
    coords0 : (2, m) array_like
        Centers of the reference patches in `img0`.
    coords1 : (2, n) array_like
        Centers of the candidate patches in `img1`.
    radius : int
        Radius of the considered patches.
    sigma : float
        Standard deviation of the Gaussian kernel centered over the patches.

    Returns:
    --------
    match_coords: (2, m) array
        The points in `coords1` that are the closest corresponding match to
        those in `coords0` as determined by the (Gaussian weighted) sum of
        squared differences between patches surrounding each point.

    """
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    weights = np.exp(-0.5 * (x ** 2 + y ** 2) / sigma ** 2)
    weights /= 2 * np.pi * sigma * sigma

    match_list = []
    for r0, c0 in coords0:
        roi0 = img0[r0 - radius:r0 + radius + 1, c0 - radius:c0 + radius + 1]
        roi1_list = [img1[r1 - radius:r1 + radius + 1,
                          c1 - radius:c1 + radius + 1] for r1, c1 in coords1]
        # sum of squared differences
        ssd_list = [np.sum(weights * (roi0 - roi1) ** 2) for roi1 in roi1_list]
        match_list.append(coords1[np.argmin(ssd_list)])

    return np.array(match_list)


############################################################################
# For this example, a set of slightly tilted noisy images are generated

img = moon()

angle_list = [5, 6, -2, 3, -4]
center_list = [(10, 10), (5, 12), (11, 21), (21, 17), (43, 15)]

i0, j0, i1, j1 = 40, 50, 240, 350
ref_img = img_as_float(img[i0:i1, j0:j1])
img_list = [ref_img.copy()] + [rotate(img, angle=a, center=c)[i0:i1, j0:j1]
                               for a, c in zip(angle_list, center_list)]

sigma = 0.015
img_list = [random_noise(gaussian(im, 1.2), var=sigma ** 2, seed=12)
            for im in img_list]

psnr_ref = peak_signal_noise_ratio(ref_img, img_list[0])

############################################################################
# Reference points are detected over all the set images
corner_list = [corner_peaks(corner_harris(img), threshold_rel=0.001,
                            min_distance=5)
               for img in img_list]

############################################################################
# The Harris corner detected in the first image are choosen as
# references. Then the detected points on the other images are
# matched to the reference points.

img0 = img_list[0]
coords0 = corner_list[0]
matching_corners = [match_locations(img0, img1, coords0, coords1)
                    for img1, coords1 in zip(img_list[1:], corner_list[1:])]

############################################################################
# Once all the points are registred to the reference points, robust
# relative affine transformations can be estimated using RANSAC method
src = np.array(coords0)
trfm_list = [ransac((dst, src), AffineTransform, min_samples=3,
                    residual_threshold=2, max_trials=100)[0].params
             for dst in matching_corners]

matching_corners = [coords0] + matching_corners
trfm_list = [np.eye(3)] + trfm_list

fig, ax_list = plt.subplots(6, 2, figsize=(4, 6), sharex=True, sharey=True)
for idx, (im, trfm, (ax0, ax1)) in enumerate(zip(img_list, trfm_list, ax_list)):
    ax0.imshow(im, cmap="gray", vmin=0, vmax=1)
    ax1.imshow(warp(im, trfm), cmap="gray", vmin=0, vmax=1)

    if idx == 0:
        ax0.set_title(f"Input (PSNR={psnr_ref:.2f})")
        ax1.set_title(f"Registered")

    ax0.set_axis_off()
    ax1.set_axis_off()

fig.tight_layout()

############################################################################
# A composite image can be obtained using the relative positions of
# the registered images to the reference one. To do so, we can define a
# global domain around the reference image and position the other
# images in this domain:
#
# A global transformation is defined to move the reference image in the
# global domain image via a simple translation
margin = 50
height, width = img_list[0].shape
out_shape = height + 2 * margin, width + 2 * margin
glob_trfm = np.eye(3)
glob_trfm[:2, 2] = -margin, -margin

############################################################################
# Now, the relative position of the other images in the global domain
# are obtained by composing the global transformation with the
# relative transformations
global_img_list = [warp(img, trfm.dot(glob_trfm), output_shape=out_shape,
                        mode="constant", cval=np.nan)
                   for img, trfm in zip(img_list, trfm_list)]

composit_img = np.nanmean(global_img_list, 0)
psnr_composit = peak_signal_noise_ratio(ref_img,
                                        composit_img[margin:margin + height,
                                                     margin:margin + width])

fig, ax = plt.subplots(1, 1)

ax.imshow(composit_img, cmap="gray", vmin=0, vmax=1)
ax.set_axis_off()
ax.set_title(f"Reconstructed image (PSNR={psnr_composit:.2f})")

fig.tight_layout()

plt.show()
