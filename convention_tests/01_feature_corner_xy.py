import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import structure_tensor, hessian_matrix

# 1. Structure Tensor with order='xy'
# Create an image varying only in X (columns)
# Image size (row, col) = (50, 100)
rows, cols = 50, 100
y, x = np.mgrid[:rows, :cols]
img_x_gradient = x.astype(float)

# Compute structure tensor with order='xy'
# Expect Axx to be non-zero, Ayy to be zero.
# If it were 'rc', the first element would be Arr (row-derivative), which should be zero.
Axx, Axy, Ayy = structure_tensor(img_x_gradient, sigma=1, order='xy')

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img_x_gradient, cmap='gray')
axes[0].set_title('Input: X-Gradient')
axes[1].imshow(Axx, cmap='gray')
axes[1].set_title('Axx (Should be high)')
axes[2].imshow(Ayy, cmap='gray')
axes[2].set_title('Ayy (Should be 0)')
axes[3].text(0.1, 0.5, f"Mean Axx: {Axx.mean():.2f}\nMean Ayy: {Ayy.mean():.2f}")
axes[3].axis('off')
plt.suptitle("structure_tensor(order='xy')")
plt.savefig('convention_tests/01_structure_tensor_xy.png')
print("Generated convention_tests/01_structure_tensor_xy.png")

# 2. Hessian Matrix with order='xy'
# Create an image with curvature in Y (rows)
# I = y^2
img_y_curvature = y.astype(float) ** 2

# Compute Hessian with order='xy'
# Hxx corresponds to d2I/dx2 (should be 0)
# Hyy corresponds to d2I/dy2 (should be constant 2)
# Hxy corresponds to d2I/dxdy (should be 0)
Hxx, Hxy, Hyy = hessian_matrix(img_y_curvature, sigma=1, order='xy')

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img_y_curvature, cmap='gray')
axes[0].set_title('Input: Y-Curvature (y^2)')
axes[1].imshow(Hxx, cmap='gray')
axes[1].set_title('Hxx (Should be 0)')
axes[2].imshow(Hyy, cmap='gray')
axes[2].set_title('Hyy (Should be > 0)')
axes[3].text(0.1, 0.5, f"Mean Hxx: {Hxx.mean():.2f}\nMean Hyy: {Hyy.mean():.2f}")
axes[3].axis('off')
plt.suptitle("hessian_matrix(order='xy')")
plt.savefig('convention_tests/01_hessian_matrix_xy.png')
print("Generated convention_tests/01_hessian_matrix_xy.png")
