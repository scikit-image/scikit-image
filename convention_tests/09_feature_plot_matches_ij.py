import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import plot_matched_features

# Test Plot Matched Features (IJ convention)
# This function PLOTS. It takes keypoints as (row, col).
# Internally it plots scatter(col, row).

rows, cols = 100, 100
img1 = np.zeros((rows, cols))
img2 = np.zeros((rows, cols))

# Create keypoints.
# Keypoint 1: (row=20, col=80) in img1
# Keypoint 2: (row=20, col=80) in img2
kp1 = np.array([[20, 80]])
kp2 = np.array([[20, 80]])
matches = np.array([[0, 0]])

fig, ax = plt.subplots(figsize=(10, 5))
plot_matched_features(
    img1, img2, keypoints0=kp1, keypoints1=kp2, matches=matches, ax=ax
)

# If it uses (row, col), the point should appear at y=20.
# If it treated input as (x, y), it would appear at y=80.
ax.set_title(
    "plot_matched_features (Input: [20, 80])\nShould be at y=20 (row), x=80 (col)"
)
ax.axhline(20, color='r', linestyle='--', label='y=20')
ax.legend()

plt.savefig('convention_tests/09_feature_plot_matches_ij.png')
print("Generated convention_tests/09_feature_plot_matches_ij.png")
