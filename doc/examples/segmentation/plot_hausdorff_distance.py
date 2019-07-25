"""
==================
Hausdorff Distance
==================

This example shows how to calculate the Hausdorff distance between two sets of
points. The `Hausdorff distance
<https://en.wikipedia.org/wiki/Hausdorff_distance>`__ is the maximum distance
between any point on the first set and its nearest point on the second set,
and vice-versa.

"""
import matplotlib.pyplot as plt
import numpy as np

from skimage import measure

shape = (600, 600)
image = np.zeros(shape)

# Create a diamond-like shape where the four corners form the 1st set of points
x_diamond = 300
y_diamond = 300
r = 100

fig, ax = plt.subplots()
plt_x = [0, 1, 0, -1, 0]
plt_y = [1, 0, -1, 0, 1]

set_ax = [(x_diamond + r*x) for x in plt_x]
set_ay = [(y_diamond + r*y) for y in plt_y]
plt.plot(set_ax, set_ay, 'r')

# Create a kite-like shape where the four corners form the 2nd set of points
x_kite = 300
y_kite = 300
x_r = 150
y_r = 200

set_bx = [(x_kite + x_r*x) for x in plt_x]
set_by = [(y_kite + y_r*y) for y in plt_y]
plt.plot(set_bx, set_by, 'b')

# Set up the data to compute the hausdorff distance
coords_a = np.zeros(shape, dtype=np.bool)
coords_b = np.zeros(shape, dtype=np.bool)
for x, y in zip(set_ax, set_ay):
    coords_a[(x, y)] = True

for x, y in zip(set_bx, set_by):
    coords_b[(x, y)] = True

# Call the hausdorff function on the coordinates
measure.set_metrics.hausdorff_distance(coords_a, coords_b)

# Plot (one of) the lines that shows the length of the hausdorff distance
x_line = [200, 300]
y_line = [300, 100]
plt.plot(x_line, y_line, 'g')

ax.imshow(image, cmap=plt.cm.gray)
ax.axis((0, 600, 600, 0))
plt.show()
