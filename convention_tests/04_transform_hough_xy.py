import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line

# Create image with a vertical line at x=20 (col 20)
# and a horizontal line at y=40 (row 40).
rows, cols = 60, 60
image = np.zeros((rows, cols), dtype=np.uint8)
image[:, 20] = 1  # Vertical line at x=20
image[40, :] = 1  # Horizontal line at y=40

# 1. Standard Hough
h, theta, d = hough_line(image)
# hough_line returns (H, theta, distances)
# We can find peaks.
_, angles, dists = hough_line_peaks(h, theta, d)

# Theta = 0 corresponds to vertical line (normal is horizontal).
# Distance should be x=20.
# Theta = pi/2 corresponds to horizontal line (normal is vertical).
# Distance should be y=40.

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Input: Lines at x=20, y=40')

axes[1].imshow(
    np.log(1 + h),
    extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
    cmap='gray',
    aspect=1 / 1.5,
)
axes[1].set_title('Hough Transform')
axes[1].set_xlabel('Angles (degrees)')
axes[1].set_ylabel('Distance (pixels)')

# Annotate peaks
for angle, dist in zip(angles, dists):
    deg = np.rad2deg(angle)
    axes[1].plot(deg, dist, 'ro')
    axes[1].text(deg, dist, f"{deg:.0f}°, {dist:.0f}px")

# 2. Probabilistic Hough
# Returns lines as ((x0, y0), (x1, y1))
lines = probabilistic_hough_line(image, threshold=10, line_length=5, line_gap=3)

axes[2].imshow(image, cmap='gray')
axes[2].set_title('Probabilistic Hough Lines')
for line in lines:
    p0, p1 = line
    # p0 is (x, y)
    axes[2].plot((p0[0], p1[0]), (p0[1], p1[1]), 'r-')
    axes[2].plot(p0[0], p0[1], 'g.')
    axes[2].plot(p1[0], p1[1], 'g.')

plt.suptitle("Hough Transform (XY convention)")
plt.savefig('convention_tests/04_transform_hough_xy.png')
print("Generated convention_tests/04_transform_hough_xy.png")
