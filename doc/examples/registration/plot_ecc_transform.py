"""
=============================================
Enhanced Correlation Coefficient Maximization
=============================================

This example illustrates how find_transform_ECC is able to reconstructe a transformation from a warped image and a template.
The code in ecc_transform is based on the OpenCV implementation FindTransformECC.
Which is itself based on the following papers:
- G. D. Evangelidis , E. Z. Psarakis, "Parametric Image Alignment using Enhanced Correlation Coefficient Maximization", IEEE Trans. Pattern Anal. Mach. Intell., 30, 10, pp. 1858-1865, 2008.
- G. D. Evangelidis E. Z. Psarakis, "Projective Image Alignment by using ECC Maximization", in Proc. Int. Conf. on Computer Vision Theory and Applications (VISSAP), January 2008, Madeira, Portugal.
"""

import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage.registration import find_transform_ECC
from skimage.transform import AffineTransform, warp

template = data.camera().astype(np.float32)

warp_matrix = AffineTransform(scale=[1, 1], rotation=np.deg2rad(5), shear=0.1, translation=[10, 15])
distorted = warp(template, warp_matrix)

rho, estimated_warp = find_transform_ECC(
    distorted, template.astype(np.float32), motion_type="MOTION_HOMOGRAPHY", number_of_iterations=200
)

corrected = warp(distorted, np.linalg.inv(estimated_warp))

fig, ax = plt.subplots(1, 3, figsize=(16, 9))

ax[0].imshow(template, cmap="gray")
ax[1].imshow(distorted, cmap="gray")
ax[2].imshow(corrected, cmap="gray")

plt.show()
