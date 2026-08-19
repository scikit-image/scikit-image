"""
================
Zernike Features
================

Zernike features are shape, texture descriptors of an object within a circular bbox.
Given an image, a unit circle centered over the image/object (circular bbox, pupil),
Zernike features are image weighted average of Zernike polynomials. Iterating over
many basis polynomials can capture fine, high-frequency details of the object.
Thus, ZFs can describe shape, structure, texture of the object.

The design parameters for extracting ZFs for an image or object are: the highest ``degree``
of the basis polynomial to use, and the ``radius``, ``center_coord`` of the circular bbox.

The example below shows an input grayscale image with noisy background and a
white square object. This object can be described by Zerike features. When computing
conventional or pseudo ZFs, a grayscale image can be used only if ``radius`` and
``center_coord`` are provided by the user, as shown in the center image of plot.
When computing ZFs where ``radius`` and ``center_coord`` are ``"auto"`` and to be
computed internally, the image needs to be binary so as to fit a convex hull and
compute the Feret diameter, centroid of the object.

A basic introductory use of ZFs can be found on `this blog
<https://cvexplained.wordpress.com/2020/07/21/10-5-zernike-moments/>`__.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import zernike_features

IMG_SIZE = 256


def create_circlebbox(center, radius):
    """Create a pupil circle for visualization."""
    y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    circle_mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= (radius**2)
    return circle_mask


rng = np.random.default_rng()

img = rng.integers(low=0, high=125, size=(IMG_SIZE, IMG_SIZE))
img[77:178, 77:178] = 255

feature_type = "default"
degree = 5
radius = 90
center = np.array([120, 120])
return_complex_moments = False

# conventional ZFs
znres = zernike_features(
    image=img,
    feature_type=feature_type,
    degree=degree,
    radius=radius,
    center_coord=center,
    return_complex_moments=return_complex_moments,
)
print("Conventional ZFs with given radius and center:")
print(znres)

imgpzf = copy.deepcopy(img)
imgpzf[img < 151] = 0
feature_type = "pseudo"
degree = 5
radius = "auto"
center = "auto"
return_complex_moments = False

# pseudo-ZFs
pznres = zernike_features(
    image=imgpzf,
    feature_type=feature_type,
    degree=degree,
    radius=radius,
    center_coord=radius,
    return_complex_moments=return_complex_moments,
)
print("\nPseudo-ZFs with computed radius and center:")
print(pznres)

imgzf = copy.deepcopy(img)
cmsk = create_circlebbox(znres.center_coord, znres.radius)
imgzf[cmsk] += 75
imgzf[img == 255] = 255

imgpzf = copy.deepcopy(img)
imgpzf[img < 151] = 0
cmsk = create_circlebbox(pznres.center_coord, pznres.radius)
imgpzf[cmsk] = 175
imgpzf[img == 255] = 255

fig, axes = plt.subplots(1, 3, figsize=(8, 4))
ax = axes.ravel()

ax[0].set_title("Grayscale image with\nbackground and object")
ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_axis_off()

ax[1].set_title("Grayscale image with\nhighlighted pupil/bbox")
ax[1].imshow(imgzf, cmap=plt.cm.gray)
ax[1].text(
    znres.center_coord[0],
    znres.center_coord[1],
    f"C=({znres.center_coord.round()}),\nR={znres.radius}",
)
ax[1].set_axis_off()

ax[2].set_title("Binary image with\ncomputed pupil/bbox")
ax[2].imshow(imgpzf, cmap=plt.cm.gray)
ax[2].text(
    pznres.center_coord[0],
    pznres.center_coord[1],
    f"C=({pznres.center_coord.round()}),\nR={pznres.radius}",
)
ax[2].set_axis_off()

plt.tight_layout()
plt.show()
