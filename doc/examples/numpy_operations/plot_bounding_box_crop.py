"""
=================================
Crop arrays by bounding boxes
=================================

This example demonstrates :py:func:`skimage.util.bounding_box_crop` to extract
regions of interest from images using spatial min/max bounds. The bounding box
values may be floats (mins are floored; maxes are ceiled; the stop is
exclusive, per Python slicing). Optionally, a ``channel_axis`` can be
specified and is not cropped.

In the figures below, the requested bbox is drawn in red (even if partially
outside the image) while the effective integer bbox actually used for slicing
is drawn in green.

See also
--------
- :py:func:`skimage.util.crop` for cropping by explicit widths on each axis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import skimage as ski


# Example 1: Grayscale image (no channel axis)
image = ski.data.camera()
bbox = ((60.3, 70.7), (280.2, 360.1))
roi = ski.util.bounding_box_crop(image, bbox)

fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharex=False, sharey=False)
ax = axes.ravel()

r0, c0 = bbox[0]
r1, c1 = bbox[1]
rect_req = Rectangle(
    (c0, r0), c1 - c0, r1 - r0, edgecolor='red', facecolor='none', linewidth=2
)
r0_i, c0_i = np.floor([r0, c0]).astype(int)
r1_i, c1_i = np.ceil([r1, c1]).astype(int)
rect = Rectangle(
    (c0_i, r0_i),
    c1_i - c0_i,
    r1_i - r0_i,
    edgecolor='lime',
    facecolor='none',
    linewidth=2,
)
ax[0].imshow(image, cmap='gray')
ax[0].add_patch(rect_req)
ax[0].add_patch(rect)
ax[0].set_title('Original with bounding box')
ax[0].set_axis_off()

ax[1].imshow(roi, cmap='gray')
ax[1].set_title('Cropped ROI (floored/ceiled bounds)')
ax[1].set_axis_off()

fig.tight_layout()


# Example 2: Color image with a channel axis
color = ski.data.astronaut()  # (M, N, 3)
bbox_color = ((120.5, 180.2), (330.0, 420.8))
roi_color = ski.util.bounding_box_crop(color, bbox_color, channel_axis=-1)

fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4), sharex=False, sharey=False)
ax2 = axes2.ravel()

r0c, c0c = bbox_color[0]
r1c, c1c = bbox_color[1]
r0c_i, c0c_i = np.floor([r0c, c0c]).astype(int)
r1c_i, c1c_i = np.ceil([r1c, c1c]).astype(int)
rect2 = Rectangle(
    (c0c_i, r0c_i),
    c1c_i - c0c_i,
    r1c_i - r0c_i,
    edgecolor='lime',
    facecolor='none',
    linewidth=2,
)
ax2[0].imshow(color)
rect2_req = Rectangle(
    (c0c, r0c), c1c - c0c, r1c - r0c, edgecolor='red', facecolor='none', linewidth=2
)
ax2[0].add_patch(rect2_req)
ax2[0].add_patch(rect2)
ax2[0].set_title('Original RGB with bounding box')
ax2[0].set_axis_off()

ax2[1].imshow(roi_color)
ax2[1].set_title('Cropped RGB ROI (channel_axis=-1)')
ax2[1].set_axis_off()

fig2.tight_layout()


# Example 3: Clipping behavior
out_of_bounds_bbox = ((-40.0, 400.0), (220.0, 700.0))
roi_clipped = ski.util.bounding_box_crop(image, out_of_bounds_bbox, clip=True)

fig3, axes3 = plt.subplots(1, 2, figsize=(8, 4), sharex=False, sharey=False)
ax3 = axes3.ravel()

r0o, c0o = out_of_bounds_bbox[0]
r1o, c1o = out_of_bounds_bbox[1]
rect3 = Rectangle(
    (c0o, r0o), c1o - c0o, r1o - r0o, edgecolor='red', facecolor='none', linewidth=2
)
ax3[0].imshow(image, cmap='gray')
ax3[0].add_patch(rect3)
ax3[0].set_title('Requested bbox (partially outside)')
ax3[0].set_axis_off()

ax3[1].imshow(roi_clipped, cmap='gray')
ax3[1].set_title('Result with clip=True')
ax3[1].set_axis_off()
ax3[1].set_anchor('C')  # ensure the ROI is centered within the right subplot

r0o_i, c0o_i = np.floor([r0o, c0o]).astype(int)
r1o_i, c1o_i = np.ceil([r1o, c1o]).astype(int)
rr0 = np.clip(r0o_i, 0, image.shape[0])
rr1 = np.clip(r1o_i, 0, image.shape[0])
cc0 = np.clip(c0o_i, 0, image.shape[1])
cc1 = np.clip(c1o_i, 0, image.shape[1])
rect3b = Rectangle(
    (cc0, rr0), cc1 - cc0, rr1 - rr0, edgecolor='lime', facecolor='none', linewidth=2
)
ax3[0].add_patch(rect3b)

fig3.tight_layout()

plt.show()
