import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import hough_line
from skimage.draw import line

img = np.zeros((100, 150), dtype=bool)
img[30, :] = 1
img[:, 65] = 1
img[35:45, 35:50] = 1
rr, cc = line(60, 130, 80, 10)
img[rr, cc] = 1
img += np.random.random(img.shape) > 0.95

out, angles, d = hough_line(img)

plt.subplot(1, 2, 1)

plt.imshow(img, cmap=plt.cm.gray)
plt.title('Input image')

plt.subplot(1, 2, 2)
plt.imshow(out, cmap=plt.cm.bone,
           extent=(np.rad2deg(angles[-1]), np.rad2deg(angles[0]),
                   d[-1], d[0]))
plt.title('Hough transform')
plt.xlabel('Angle (degree)')
plt.ylabel('Distance (pixel)')

plt.subplots_adjust(wspace=0.4)
plt.show()
