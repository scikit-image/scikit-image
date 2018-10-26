from skimage.data import camera
from scipy import ndimage as ndi
from skimage.transform import register_affine
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure

rows=6
cols=6
start_with=10
show_every=3
callback_arr = []
callback = lambda i, m: callback_arr.append((i,m))
r = 0.12
out = np.empty((3, 3))

out[0][0] = np.cos(r)
out[0][1] = -np.sin(r)
out[0][2] = 0.2
out[1][0] = np.sin(r)
out[1][1] = np.cos(r)
out[1][2] = 0.1
out[2][0] = 0
out[2][1] = 0
out[2][2] = 1

cam = camera()
start = ndi.affine_transform(cam, out)
start = ndi.shift(start, (0, 50))
trans = register_affine(cam, start, iter_callback=callback)
_, ax = plt.subplots(1, 6)

ax[0].set_title('reference')
ax[0].imshow(cam, cmap='gray')
y, x = cam.shape
ax[0].set_xticks(np.arange(x/5, x, x/5), minor=True)
ax[0].set_yticks(np.arange(y/5, y, y/5), minor=True)
ax[0].grid(which='minor', color='w', linestyle='-', linewidth=1)

err = measure.compare_mse(cam, start)
ax[1].set_title('target, mse %d' % int(err))
ax[1].imshow(start, cmap='gray')
y, x = start.shape
ax[1].set_xticks(np.arange(x/5, x, x/5), minor=True)
ax[1].set_yticks(np.arange(y/5, y, y/5), minor=True)
ax[1].grid(which='minor', color='w', linestyle='-', linewidth=1)

for a in ax:
    a.set_xticklabels([])
    a.set_yticklabels([])

for i, image in enumerate([1, 2, 4]):
    err = measure.compare_mse(cam, ndi.affine_transform(start, callback_arr[image][1]))
    ax[i+2].set_title('iter %d, mse %d' % (image, int(err)))
    ax[i+2].imshow(ndi.affine_transform(callback_arr[image][0], callback_arr[image][1]), cmap='gray',
         interpolation='gaussian', resample=True)

    y, x = callback_arr[image][0].shape

    ax[i+2].set_xticks(np.arange(x/5, x, x/5), minor=True)
    ax[i+2].set_yticks(np.arange(y/5, y, y/5), minor=True)
    ax[i+2].grid(which='minor', color='w', linestyle='-', linewidth=1)

err = measure.compare_mse(cam, ndi.affine_transform(start, trans))
ax[5].set_title('final correction, mse %d' % int(err))
ax[5].imshow(ndi.affine_transform(start, trans),cmap='gray',
        interpolation='gaussian', resample=True)
y, x = start.shape

ax[5].set_xticks(np.arange(x/5, x, x/5), minor=True)
ax[5].set_yticks(np.arange(y/5, y, y/5), minor=True)
ax[5].grid(which='minor', color='w', linestyle='-', linewidth=1)
plt.show()