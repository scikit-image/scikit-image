from skimage.segmentation import graphcut
import numpy as np
import matplotlib.pyplot as plt

import Image
from skimage import data_dir

data = Image.open(data_dir+'/'+'trolls.png')
if data.mode != 'RGBA':
    data = data.convert('RGBA')

img = np.array(data)
src = np.load(data_dir+'/'+'trolls_fg.npy').reshape((608, 800))[0:600, :]
sink = np.load(data_dir+'/'+'/trolls_bg.npy').reshape((608, 800))[0:600, :]

gc = graphcut(img, src, sink, 10, 60)
plt.imshow(gc, interpolation='nearest')
plt.show()

True