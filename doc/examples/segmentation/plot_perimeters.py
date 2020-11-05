"""
=========================
Different perimeters
=========================

In this example we show the uncertainty on calculating perimeters, comparing
classic and Crofton ones. For that, we evaluate the perimeters of a square and
its rotated version.

"""
from skimage.measure import perimeter
from skimage.measure import perimeter_crofton
from skimage.transform import rotate
import matplotlib.pyplot as plt
import numpy as np


# scale parameter can be used to increase the grid size. The resulting curves
# should be smoothed with higer scales
scale = 10

# Construct 2 figures, square and disks
square = np.zeros((100*scale, 100*scale))
square[40*scale:60*scale, 40*scale:60*scale] = 1

[X, Y] = np.meshgrid(np.linspace(0, 100*scale), np.linspace(0, 100*scale))
R = 20 * scale
disk = (X-50*scale)**2+(Y-50*scale)**2 <= R**2

fig, axes = plt.subplots(1, 2, figsize=(8, 5))
ax = axes.flatten()

dX = X[0, 1] - X[0, 0]
true_perimeters = [80 * scale, 2 * np.pi * R / dX]

# for each type of objects, the different perimeters are evaluated
for index, obj in enumerate([square, disk]):

    # 2 neighbourhoud configurations for measure.perimeter
    for n in [4, 6]:
        p = []
        angles = range(90)
        for i in angles:
            # rotation and perimeter evaluation
            rotated = rotate(obj, i, order=0)
            p.append(perimeter(rotated, n))
        ax[index].plot(angles, p)

    # 2 or 4 directions can be used by measure.perimeter_crofton
    for d in [2, 4]:
        p = []
        angles = np.arange(0, 90, 2)
        for i in angles:
            # rotation and perimeter evaluation
            rotated = rotate(obj, i, order=0)
            p.append(perimeter_crofton(rotated, d))
        ax[index].plot(angles, p)

    ax[index].axhline(true_perimeters[index], linestyle='--', color='k')
    ax[index].set_xlabel('Rotation angle')
    ax[index].legend(['N4 perimeter', 'N8 perimeter',
                      'Crofton 2 directions', 'Crofton 4 directions',
                      'Ground truth'],
                      loc='best')
    ax[index].set_ylabel('Perimeter of the rotated object')

ax[0].set_title('Square')
ax[1].set_title('Disk')
plt.tight_layout()
plt.show()
