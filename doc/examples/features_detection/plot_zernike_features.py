"""
================
Zernike Features
================

Zernike features are shape, texture descriptors of an object within a given pupil.
Given an image, a unit circle pupil centered over the image/object, Zernike features
are image weighted average of basis Zernike polynomials. Iterating over many basis
polynomials can capture fine, high-frequency details of the object. Thus, ZFs can
describe shape, structure, texture of the object.

Zernike polynomials are of two types: conventional, which are regular Zernike polynomials
defined for even modes, and pseudo, which are pseudo-Zernike polynomials which considers
all modes and are denser (more) descriptors per image.

The design parameters for extracting ZFs for an image or object are: the highest ``degree``
of the basis polynomial to use, and the pupil parameters ``primary_dim`` , ``secondary_dim`` ,
``center_coord`` of the pupil. There are 6 pupil shapes available: circle, annulus,
ellipse. rectangle, square, regular hexagon. The parameters ``primary_dim`` , ``secondary_dim``
describe the size or dimensions of the pupil like radius, length, width, side, axes etc.

For the examples below, consider an input grayscale image with noisy background and a
white square object. This object can be described by Zerike features. The examples are:

- Basic use of the ``zernike_features`` API using conventional-ZFs and a given pupil, and automated calculation of ``primary_dim`` , ``secondary_dim`` , ``center_coord`` pupil parameters for pseudo-ZFs.
- All 6 pupil shapes shown over object as given by user, or computed automatically.
- Reconstruction of given image using computed ZFs.
- Finding optimum ``degree`` parameter by running a sweep over multiple values and measuring SSIM (structure similarity index measure) over input image and reconstructed image.

A basic introductory use of ZFs can be found on `this blog
<https://cvexplained.wordpress.com/2020/07/21/10-5-zernike-moments/>`__.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import zernike_features

IMG_SIZE = 256


def get_znfts(img, ft, pt, deg, pd, sd, cc, rcm, rpm, rri):
    znres = zernike_features(
        image=img,
        feature_type=ft,
        pupil_type=pt,
        degree=deg,
        primary_dim=pd,
        secondary_dim=sd,
        center_coord=cc,
        return_complex_moments=rcm,
        return_pupil_mask=rpm,
        return_reconstructed_image=rri,
    )
    return znres


rng = np.random.default_rng()

img = rng.integers(low=0, high=125, size=(IMG_SIZE, IMG_SIZE))
img[77:178, 77:178] = 255  # square object
# img[77:178, 57:188] = 255  # wide object
# img[60:141, 80:121] = 255  # tall object

# compile basic examples
# conventional ZFs
znresc = get_znfts(
    img, "conventional", "circle", 5, 90, None, np.array([120, 120]), False, True, True
)
print("Conventional ZFs with given radius and center:")
print(znresc)

imgpzf = copy.deepcopy(img)
imgpzf[img < 151] = 0

# pseudo-ZFs
znresp = get_znfts(
    imgpzf, "pseudo", "circle", 5, "auto", None, "auto", False, True, True
)

print("\nPseudo-ZFs with computed radius and center:")
print(znresp)

imgzf = copy.deepcopy(img)
cmsk = znresc.pupil_mask
imgzf[cmsk] += 75
imgzf[img == 255] = 255

imgpzf = copy.deepcopy(img)
imgpzf[img < 151] = 0
cmsk = znresp.pupil_mask
imgpzf[cmsk] = 175
imgpzf[img == 255] = 255

fig, axes = plt.subplots(1, 3, figsize=(8, 4))
ax = axes.ravel()

paired_data = (
    (img, None, "Grayscale image with\nbackground and object"),
    (imgzf, znresc, "Grayscale image with given\npupil/bbox, highlighted."),
    (imgpzf, znresp, "Binary image with\ncomputed pupil/bbox"),
)
for idx, op in enumerate(paired_data):
    (
        imgo,
        zno,
        ttl,
    ) = op
    ax[idx].set_title(ttl)
    ax[idx].imshow(imgo, cmap=plt.cm.gray)
    if zno is not None:
        ax[idx].text(
            zno.center_coord[0],
            zno.center_coord[1],
            f"C={zno.center_coord.round()},\nPD={zno.primary_dim}",
        )
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()

# compile pupil plots
paired_data = (
    ("circle", 90, None, np.array([120, 120]), "Circular pupil\nuser-defined"),
    ("circle", "auto", None, "auto", "Circular pupil\nautomated"),
    ("annulus", 90, 60.0, np.array([120, 120]), "Annular pupil\nuser-defined"),
    ("annulus", "auto", 0.8, "auto", "Annular pupil\nautomated"),
    ("ellipse", 90, 30.0, np.array([120, 120]), "Elliptical pupil\nuser-defined"),
    ("ellipse", "auto", None, "auto", "Elliptical pupil\nautomated"),
    ("rectangle", 90, 30.0, np.array([120, 120]), "Rectangular pupil\nuser-defined"),
    ("rectangle", "auto", None, "auto", "Rectangular pupil\nautomated"),
    ("square", 90, None, np.array([120, 120]), "Square pupil\nuser-defined"),
    ("square", "auto", None, "auto", "Square pupil\nautomated"),
    ("hexagon", 90, None, np.array([120, 120]), "Hexagonal pupil\nuser-defined"),
    ("hexagon", "auto", None, "auto", "Hexagonal pupil\nautomated"),
)
fig, axes = plt.subplots(3, 4, figsize=(8, 4))
ax = axes.ravel()
for idx, onepair in enumerate(paired_data):
    pt, pd, sd, cc, ttl = onepair
    imgo = copy.deepcopy(img)

    if pd == "auto":
        imgo[img < 151] = 0

    zno = get_znfts(imgo, "conventional", pt, 5, pd, sd, cc, False, True, False)
    cmsk = zno.pupil_mask

    imgo = copy.deepcopy(img)
    imgo[cmsk] += 75

    ax[idx].set_title(ttl)
    ax[idx].imshow(imgo, cmap=plt.cm.gray)
    ax[idx].text(
        zno.center_coord[0],
        zno.center_coord[1],
        f"PD={zno.primary_dim}\nSD={zno.secondary_dim}",
    )
    ax[idx].set_axis_off()
plt.tight_layout()
plt.show()

# compile reconstruction images
paired_data = (
    (
        img,
        "conventional",
        "circle",
        None,
        90,
        None,
        np.array([120, 120]),
        "Original image",
    ),
    (
        img,
        "conventional",
        "circle",
        2,
        90,
        None,
        np.array([120, 120]),
        "Reconstructed image\nconventional zfs\ncircle pupil\nuser-defined\ndegree=2",
    ),
    (
        img,
        "conventional",
        "circle",
        30,
        90,
        None,
        np.array([120, 120]),
        "Reconstructed image\nconventional zfs\ncircle pupil\nuser-defined\ndegree=30",
    ),
    (img, "pseudo", "square", None, "auto", None, "auto", "Original image"),
    (
        img,
        "pseudo",
        "square",
        2,
        "auto",
        None,
        "auto",
        "Reconstructed image\npseudo zfs\nsquare pupil\nautomated\ndegree=2",
    ),
    (
        img,
        "pseudo",
        "square",
        30,
        "auto",
        None,
        "auto",
        "Reconstructed image\npseudo zfs\nsquare pupil\nautomated\ndegree=30",
    ),
)
fig, axes = plt.subplots(2, 3, figsize=(8, 4))
ax = axes.ravel()
for idx, onepair in enumerate(paired_data):
    imgo, ft, pt, deg, pd, sd, cc, ttl = onepair
    if deg is not None:
        if pd == "auto":
            imgo[img < 151] = 0
        zno = get_znfts(imgo, ft, pt, deg, pd, sd, cc, False, False, True)
        imgo = zno.reconstructed_image
    ax[idx].set_title(ttl)
    ax[idx].imshow(imgo, cmap=plt.cm.gray)
    ax[idx].set_axis_off()
plt.tight_layout()
plt.show()

# compile optimized degree plot
degrees = [x for x in range(2, 31)]
mse_circle = []
mse_square = []
for deg in degrees:
    znoc = get_znfts(
        img,
        "conventional",
        "circle",
        deg,
        90,
        None,
        np.array([120, 120]),
        False,
        False,
        True,
    )
    imgo = copy.deepcopy(img)
    imgo[img < 151] = 0
    znos = get_znfts(
        imgo, "pseudo", "square", deg, "auto", None, "auto", False, False, True
    )
    msec = np.mean((img - znoc.reconstructed_image) ** 2.0)
    mse_circle.append(msec.item())
    mses = np.mean((img - znos.reconstructed_image) ** 2.0)
    mse_square.append(mses.item())

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(
    degrees,
    mse_circle,
    label="MSE-Conv.-Circle-User",
    color="blue",
    marker="o",
    linestyle="-",
)
ax.plot(
    degrees,
    mse_square,
    label="MSE-Pseudo-Square-Auto",
    color="red",
    marker="x",
    linestyle="--",
)

plt.xlabel("Degrees")
plt.ylabel("Mean-Squared Error (MSE)")
plt.title("Optimal degree using elbow plot.")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
