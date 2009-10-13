import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import AxesGrid
import numpy as np

from scikits.image.io import MultiImage
from scikits.image import data_dir


fname = os.path.join(data_dir, 'multipage.tif')
img = MultiImage(fname)

fig = plt.figure()
grid = AxesGrid(fig, 111,
                nrows_ncols = (1, 2),
                axes_pad = 0.1,
                add_all=True,
                label_mode = "L",
                aspect=True)

for i, frame in enumerate(img):
    grid[i].imshow(frame, cmap=plt.cm.gray)
    grid[i].set_xlabel('Frame %s'%i)
    grid[i].set_xticks([])
    grid[i].set_yticks([])

plt.show()
