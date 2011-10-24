import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import AxesGrid

from skimage.io import MultiImage
from skimage import data_dir

# Load the multi-layer image
fname = os.path.join(data_dir, 'multipage.tif')
img = MultiImage(fname)

# Create an image grid
fig = plt.figure()
grid = AxesGrid(fig,
                rect=(1, 1, 1),
                nrows_ncols=(1, 2),
                axes_pad=0.1)

# Plot the layers on the image grid
for i, frame in enumerate(img):
    grid[i].imshow(frame, cmap=plt.cm.gray)
    grid[i].set_xlabel('Frame %s' % i)
    grid[i].set_xticks([])
    grid[i].set_yticks([])

plt.show()
