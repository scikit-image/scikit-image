from skimage.segmentation import growcut
import numpy as np
from skimage import io, img_as_float

from skimage import data_dir

image = img_as_float(io.imread(data_dir+'/sharkfin.jpg'))
state = np.zeros((image.shape[0], image.shape[1], 2))

foreground_pixels = np.array([(150, 200), (200, 300)])
background_pixels = np.array([(50, 100), (50, 400), (250, 50), (200, 400)])

for (r, c) in background_pixels:
    state[r, c] = (0, 1)

for (r, c) in foreground_pixels:
    state[r, c] = (1, 1)

out = growcut(image, state, window_size=5, max_iter=500)

import matplotlib.pyplot as plt

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(7, 3))

ax0.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
ax0.plot(foreground_pixels[:, 1], foreground_pixels[:, 0],
    color='blue', marker='o', linestyle='none', label='Foreground')
ax0.plot(background_pixels[:, 1], background_pixels[:, 0],
    color='red', marker='o', linestyle='none', label='Background')
ax0.set_title('Input image')
ax0.axis('image')

ax1.imshow(out[..., None] * image, interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)
ax1.set_title('Foreground / background')
ax0.axis('image')

plt.show()
