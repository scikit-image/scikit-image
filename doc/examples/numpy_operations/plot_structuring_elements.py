"""
=============================
Generate structuring elements
=============================

This example shows how to use functions in :py:module:`skimage.morphology`
to generate structuring elements.
The title of each plot indicates the call of the function.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star)

# Generate 2D and 3D structuring elements.
# [(title, data)]
struc_2d = [
    ("square(5)", square(5)),
    ("rectangle(5, 3)", rectangle(5, 3)),
    ("diamond(5)", diamond(5)),
    ("disk(7)", disk(7)),
    ("octagon(4, 7)", octagon(4, 7)),
    ("star(6)", star(6))
]

struc_3d = [
    ("cube(5)", cube(5)),
    ("octahedron(5)", octahedron(5)),
    ("ball(5)", ball(5)),
]

# Visualize the elements.
fig = plt.figure(figsize=(8, 8))

idx = 1
for title, struc in struc_2d:
    ax = fig.add_subplot(3, 3, idx)
    ax.imshow(struc, cmap="Paired", vmin=0, vmax=12)
    for i in range(struc.shape[0]):
        for j in range(struc.shape[1]):
            ax.text(j, i, struc[i, j], ha="center", va="center", color="w")
    ax.set_axis_off()
    ax.set_title(title)
    idx += 1

for title, struc in struc_3d:
    ax = fig.add_subplot(3, 3, idx, projection=Axes3D.name)
    x, y, z = np.where(struc == 1)
    ax.scatter(x, y, z)
    ax.set_title(title)
    idx += 1

fig.tight_layout()
plt.show()
