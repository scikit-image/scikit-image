import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.filters import gaussian

# Test Blob Detection (IJ convention)
# Create blobs at known (row, col) locations.
# Blob 1: Row=20, Col=80.
# Blob 2: Row=70, Col=30.
rows, cols = 100, 100
image = np.zeros((rows, cols))
# Make them bright enough to be detected after smoothing
image[20, 80] = 10
image[70, 30] = 10

# Blur to make them blobs
# sigma=2
image = gaussian(image, sigma=2)
# Peak value will be roughly 10 * 1/(2*pi*2^2) ~ 10/25 ~ 0.4.

# Detect
# threshold should be low enough. Default is 0.5?
blobs_dog = blob_dog(image, min_sigma=1, max_sigma=5, threshold=0.1)
blobs_log = blob_log(image, min_sigma=1, max_sigma=5, threshold=0.1)
blobs_doh = blob_doh(image, min_sigma=1, max_sigma=5, threshold=0.01)

# All return (row, col, sigma)
print("DoG:", blobs_dog)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Input: Blobs at (r=20,c=80), (r=70,c=30)')


def plot_blobs(ax, blobs, title):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    if len(blobs) > 0:
        for blob in blobs:
            r, c, sigma = blob
            # Circle takes (x, y) = (col, row)
            circ = plt.Circle((c, r), sigma * np.sqrt(2), color='r', fill=False)
            ax.add_patch(circ)
            ax.text(c, r, f"r={r:.0f},c={c:.0f}", color='yellow', fontsize=8)
    else:
        ax.text(cols // 2, rows // 2, "No blobs found", color='red', ha='center')


plot_blobs(axes[1], blobs_dog, 'Blob DoG (r, c, sigma)')
plot_blobs(axes[2], blobs_log, 'Blob LoG (r, c, sigma)')
plot_blobs(axes[3], blobs_doh, 'Blob DoH (r, c, sigma)')

plt.suptitle("Blob Detection (IJ convention)")
plt.savefig('convention_tests/07_feature_blob_ij.png')
print("Generated convention_tests/07_feature_blob_ij.png")
