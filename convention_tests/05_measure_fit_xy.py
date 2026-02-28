import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import CircleModel, EllipseModel, LineModelND

# 1. Circle Model
# Create data points on a circle at (x, y) = (30, 20) with radius 10.
t = np.linspace(0, 2 * np.pi, 50)
xc, yc = 30, 20
r = 10
x = xc + r * np.cos(t)
y = yc + r * np.sin(t)
data = np.column_stack([x, y])

model = CircleModel()
model.estimate(data)

# params: xc, yc, r
est_xc, est_yc, est_r = model.params

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot for Circle
axes[0].plot(x, y, 'b.', label='Data')
# Draw estimated circle
circ_t = np.linspace(0, 2 * np.pi, 100)
circ_x = est_xc + est_r * np.cos(circ_t)
circ_y = est_yc + est_r * np.sin(circ_t)
axes[0].plot(circ_x, circ_y, 'r-', label='Fit')
axes[0].set_title(
    f"CircleModel\nEst center=({est_xc:.1f}, {est_yc:.1f})\nTrue center=(30, 20)"
)
axes[0].set_aspect('equal')
axes[0].invert_yaxis()  # Match image coords
axes[0].legend()

# 2. Ellipse Model
# Create data points on an ellipse at (x, y) = (40, 30)
xe, ye = 40, 30
a, b = 15, 8
theta = np.deg2rad(30)
t = np.linspace(0, 2 * np.pi, 50)
x_ell = xe + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
y_ell = ye + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
data_ell = np.column_stack([x_ell, y_ell])

model_ell = EllipseModel()
model_ell.estimate(data_ell)
est_xe, est_ye, est_a, est_b, est_theta = model_ell.params

# Plot for Ellipse
axes[1].plot(x_ell, y_ell, 'b.', label='Data')
ell_x, ell_y = model_ell.predict_xy(
    np.linspace(0, 2 * np.pi, 100), params=model_ell.params
).T
axes[1].plot(ell_x, ell_y, 'r-', label='Fit')
axes[1].set_title(
    f"EllipseModel\nEst center=({est_xe:.1f}, {est_ye:.1f})\nTrue center=(40, 30)"
)
axes[1].set_aspect('equal')
axes[1].invert_yaxis()
axes[1].legend()

# 3. LineModelND
# Create data points on a line y = x + 10.
# Points (0, 10), (50, 60).
xl = np.linspace(0, 50, 20)
yl = xl + 10
data_line = np.column_stack([xl, yl])

model_line = LineModelND()
model_line.estimate(data_line)
# predict_y(x)
pred_y = model_line.predict_y(xl)

axes[2].plot(xl, yl, 'b.', label='Data')
axes[2].plot(xl, pred_y, 'r-', label='Fit')
axes[2].set_title("LineModelND\ny = x + 10")
axes[2].set_aspect('equal')
axes[2].invert_yaxis()
axes[2].legend()

plt.suptitle("Measure Fit Models (XY convention)")
plt.savefig('convention_tests/05_measure_fit_xy.png')
print("Generated convention_tests/05_measure_fit_xy.png")
