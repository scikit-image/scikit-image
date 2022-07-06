"""
=========================
Render text onto an image
=========================

Scikit-image currently doesn't feature a function that allows you to plot/render
text onto an image. However, there is a fairly easy workaround using the
`matplotlib <https://matplotlib.org/>`_ library.

"""

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

img = iio.imread("imageio:chelsea.png")

fig, ax = plt.subplots(figsize=figaspect(img), dpi=75)
ax.imshow(img)
ax.text(5, 5, "I am stefan's cat.", fontsize=32, va="top")
ax.set_axis_off()
ax.set_position([0, 0, 1, 1])
fig.canvas.draw()
annotated_img = np.asarray(fig.canvas.renderer.buffer_rgba())
plt.close(fig)


################################################################################
# For the purpose of this example, we can also show the image; however, if one
# just wants to render the image on top of the text, this step is not necessary.

fig, ax = plt.subplots()
ax.imshow(annotated_img)
ax.set_axis_off()
ax.set_position([0, 0, 1, 1])
plt.show()
