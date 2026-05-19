import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import multiblock_lbp
from skimage.transform import integral_image

# Test Multiblock LBP (IJ convention)
# We want to confirm r is row (y) and c is col (x).

rows, cols = 50, 50
image = np.full((rows, cols), 10, dtype=np.uint8)

# Set up a pattern
# Center block area for query (r=10, c=0): r=20..30, c=10..20. Value 20.
image[20:30, 10:20] = 20
# Top-Center block area: r=10..20, c=10..20. Value 30.
image[10:20, 10:20] = 30

int_img = integral_image(image)

# Query at r=10, c=0 (width=10, height=10).
# If r is row, center is at r=20, c=10. This matches our value 20.
# Top-center is at r=10, c=10. Matches 30.
# 30 >= 20 -> True.
# Others 10 >= 20 -> False.
# Result should be power of 2 (single bit set).

lbp_r_row = multiblock_lbp(int_img, r=10, c=0, width=10, height=10)

# If r was col (x), then r=10 means x=10 -> c=10.
# c=0 means y=0 -> r=0.
# So query would be at r=0, c=10.
# Grid r=0..30, c=10..40.
# Center r=10..20, c=20..30. (Value 10).
# Top-Center r=0..10, c=20..30. (Value 10).
# Left-Center r=10..20, c=10..20. (Value 30).
# So Center=10. Left-Center=30.
# 30 >= 10 -> True.
# Others 10 >= 10 -> True.
# Result would be 255 (all ones).

print(f"LBP(r=10, c=0): {lbp_r_row}")

fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
ax.set_title(
    f"Multiblock LBP Check\nBackground=10, Center=20, Top=30\nQuery(r=10, c=0)\nResult {lbp_r_row} (Expected single bit)\nIf swapped, expected 255"
)
plt.savefig('convention_tests/10_feature_texture_ij.png')
print("Generated convention_tests/10_feature_texture_ij.png")
