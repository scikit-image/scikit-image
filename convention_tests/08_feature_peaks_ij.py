import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import (
    peak_local_max,
    corner_peaks,
    corner_subpix,
    corner_orientations,
    gaussian,
)
from skimage.morphology import octagon

# Test Peaks (IJ convention)
# Create peaks at (row=20, col=80) and (row=70, col=30)
rows, cols = 100, 100
image = np.zeros((rows, cols))
image[20, 80] = 1
image[70, 30] = 1

# 1. Peak Local Max
# Returns (row, col)
peaks = peak_local_max(image, min_distance=1)
print("Peaks:", peaks)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Input: Peaks at (20, 80), (70, 30)')
axes[0].plot(peaks[:, 1], peaks[:, 0], 'rx', label='peak_local_max')  # Plot col, row
axes[0].legend()

# 2. Corner Peaks
# Use corner_harris to generate a corner response, but here we just use the image
# as a response map for simplicity since corner_peaks just finds peaks in a 2D array.
c_peaks = corner_peaks(image, min_distance=1)
axes[1].imshow(image, cmap='gray')
axes[1].set_title('corner_peaks (r, c)')
axes[1].plot(c_peaks[:, 1], c_peaks[:, 0], 'gx', label='corner_peaks')  # Plot col, row
axes[1].legend()

# 3. Corner Subpix
# Needs corners as input.
# Let's shift the peak slightly off-grid by blurring

image_sub = gaussian(image, sigma=1)
# Subpix refines (row, col)
subpix = corner_subpix(image_sub, c_peaks, window_size=5)
axes[2].imshow(image_sub, cmap='gray')
axes[2].set_title('corner_subpix (r, c)')
axes[2].plot(subpix[:, 1], subpix[:, 0], 'mx', label='subpix')
axes[2].legend()
axes[2].text(0, 10, f"Subpix 1: {subpix[0]}")

# 4. Corner Orientations
# corner_orientations(image, corners, mask)
# Returns orientations.
# We need an image with some structure to have orientation.
# Let's make a corner at (50, 50).
img_corner = np.zeros((100, 100))
img_corner[50:, 50:] = 1  # Bottom-right quadrant filled.
# Corner at (50, 50).
# Gradient points diagonal?

corners_ori = np.array([[50, 50]])
mask_ori = octagon(5, 2)
# orientations computed.
orientations = corner_orientations(img_corner, corners_ori, mask_ori)
# Orientation is angle of vector from corner to centroid of intensity in mask.
# Centroid of intensity is in bottom-right.
# So vector points +r, +c.
# In IJ: +row is down, +col is right.
# Centroid is positive row, positive col.
# Angle should be around +45 degrees (pi/4) if 0 is right?
# Wait, angle is usually atan2(y, x).
# If IJ is used as (y, x), then atan2(row, col).
# Let's check.
print(f"Orientation: {np.rad2deg(orientations[0])} deg")

axes[3].imshow(img_corner, cmap='gray')
axes[3].set_title(
    f'corner_orientations\nAngle: {np.rad2deg(np.ravel(orientations)[0]):.1f} deg'
)
axes[3].plot(50, 50, 'ro')
# Plot arrow
angle = np.ravel(orientations)[0]
# If angle is standard math angle (CCW from x), then dx=cos(a), dy=sin(a).
# But y is row (down).
axes[3].arrow(50, 50, 20 * np.cos(angle), 20 * np.sin(angle), color='y', head_width=2)

plt.suptitle("Peak/Corner Detection (IJ convention)")
plt.savefig('convention_tests/08_feature_peaks_ij.png')
print("Generated convention_tests/08_feature_peaks_ij.png")
