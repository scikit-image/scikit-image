"""
====================================================
Active Contour Model
====================================================
The active contour model is a method to fit open or closed splines to lines or
edges in an image. It works by minimising an energy that is in part defined by
the image and part by the spline's shape: length and smoothness. The
minimization is done implicitly in the shape energy and explicitly in the
image energy.

In the following two examples the active contour model is used (1) to segment
the face of a person from the rest of an image by fitting a closed curve
to the edges of the face and (2) to find the darkest curve between two fixed
points while obeying smoothness considerations. Typically it is a good idea to
smooth images a bit before analyzing, as done in the following examples.

.. [1] *Snakes: Active contour models*. Kass, M.; Witkin, A.; Terzopoulos, D.
       International Journal of Computer Vision 1 (4): 321 (1988).

We initialize a circle around the astronaut's face and use the defualt boundary
condition `bc='periodic'` to fit a closed curve. The default parameters
`w_line=0, w_edge=1` will make the curve search towards edges, such as the
boundaries of the face.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian_filter
from skimage.segmentation import active_contour_model

img = data.astronaut()
img = rgb2gray(img)

s = np.linspace(0, 2*np.pi, 400)
x = 220 + 100*np.cos(s)
y = 100 + 100*np.sin(s)
init = np.array([x, y]).T

snake = active_contour_model(gaussian_filter(img, 3),
    init, alpha=0.015, beta=10, gamma=0.001)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
plt.gray()
ax.imshow(img)
ax.plot(init[:, 0], init[:, 1], '--r')
ax.plot(snake[:, 0], snake[:, 1], '-b')
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

"""
.. image:: PLOT2RST.current_figure

Here we initialize a straight line between two points, `(5, 136)` and
`(424, 50)`, and require that the spline has its end points there by giving
the boundary condition `bc='fixed'`. We furthermore make the algorithm search
for dark lines by giving a negative `w_line` value.
"""

img = data.text()

x = np.linspace(5, 424, 100)
y = np.linspace(136, 50, 100)
init = np.array([x, y]).T

snake = active_contour_model(gaussian_filter(img, 1), init, bc='fixed',
        alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
plt.gray()
ax.imshow(img)
ax.plot(init[:, 0], init[:, 1], '--r')
ax.plot(snake[:, 0], snake[:, 1], '-b')
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()

"""
.. image:: PLOT2RST.current_figure
"""
