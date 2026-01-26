import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops

# Test Regionprops (IJ convention)
# Create a labeled image with a rectangle.
# Rows: 10..30 (height 20), Cols: 50..80 (width 30).
# Center should be Row=20, Col=65.
# In XY, this is y=20, x=65.
# But numpy indexing is [row, col].

rows, cols = 100, 100
image = np.zeros((rows, cols), dtype=int)
image[10:30, 50:80] = 1  # Label 1

props = regionprops(image)[0]
centroid = props.centroid  # (row, col)
bbox = props.bbox  # (min_row, min_col, max_row, max_col)
coords = props.coords  # (N, 2) -> (row, col)

print(f"Centroid: {centroid}")
print(f"BBox: {bbox}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image, cmap='gray')
ax.set_title(f"RegionProps (IJ)\nCentroid: {centroid} (row, col)\nBBox: {bbox}")

# Plot centroid. Note scatter takes (x, y) = (col, row).
# So we must flip centroid for plotting.
ax.scatter([centroid[1]], [centroid[0]], c='r', label='Centroid (col, row)')

# Plot Bbox
minr, minc, maxr, maxc = bbox
rect = plt.Rectangle(
    (minc, minr),
    maxc - minc,
    maxr - minr,
    fill=False,
    edgecolor='blue',
    linewidth=2,
    label='BBox',
)
ax.add_patch(rect)

# Plot first few coords to verify
# coords are (row, col). Plot as (col, row)
ax.plot(coords[:10, 1], coords[:10, 0], 'g.', markersize=2, label='Coords (sample)')

ax.legend()
plt.savefig('convention_tests/06_measure_regionprops_ij.png')
print("Generated convention_tests/06_measure_regionprops_ij.png")
