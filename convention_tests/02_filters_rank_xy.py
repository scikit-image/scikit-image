import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import mean
from skimage.morphology import disk

# Test Rank Filters (mean) with shift_x, shift_y
# Create an impulse image
rows, cols = 51, 51
image = np.zeros((rows, cols), dtype=np.uint8)
# Center impulse
image[rows // 2, cols // 2] = 255

# Use a larger disk to allow for larger shifts.
# disk(5) has shape (11, 11). Center at (5, 5).
# If we shift by 5, center becomes (5, 10) or (10, 5) which is within bounds [0, 10].
footprint = disk(5)
shift_amt = 5

# Apply mean filter with shift_x = 5 (positive x is right)
# shift_x adds to the column coordinate of the center.
# If center is shifted +5 (right), the filter "looks" 5 pixels to the right.
# So the output at (r, c) sees (r, c+5).
# The impulse at (25, 25) will be seen by (25, 20).
# So the feature should shift LEFT (to col 20).
out_x = mean(image, footprint, shift_x=shift_amt, shift_y=0)

# Shift_y = 5.
# Feature should shift UP (to row 20).
out_y = mean(image, footprint, shift_x=0, shift_y=shift_amt)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Input Impulse (25, 25)')
axes[0].axvline(25, color='r', alpha=0.5)
axes[0].axhline(25, color='r', alpha=0.5)

axes[1].imshow(out_x, cmap='gray')
axes[1].set_title(f'shift_x={shift_amt}\n(Object shifts LEFT)')
axes[1].axvline(25 - shift_amt, color='r', alpha=0.5)

axes[2].imshow(out_y, cmap='gray')
axes[2].set_title(f'shift_y={shift_amt}\n(Object shifts UP)')
axes[2].axhline(25 - shift_amt, color='r', alpha=0.5)

plt.suptitle("rank.mean(shift_x, shift_y)")
plt.savefig('convention_tests/02_filters_rank_xy.png')
print("Generated convention_tests/02_filters_rank_xy.png")
