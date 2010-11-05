import os

import scikits.image as si
import scikits.image.io as sio

sio.use_plugin('matplotlib', 'imshow')
sio.use_plugin('freeimage', 'imread')

img = sio.imread(os.path.join(si.data_dir, 'color.png'))

sio.imshow(img)
sio.show()

