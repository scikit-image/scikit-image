import numpy as np
import Image
from skimage import data_dir
from skimage.segmentation import graphcut
import matplotlib.pyplot as plt
import time

cmap_gray = plt.get_cmap('gray')

cmap_jet = plt.get_cmap('jet')
cmap_jet.set_under(color='k')
cmap_jet.set_over(color='w')

test_gray = False

src = np.load(data_dir+'/'+'trolls_prob_fg_gen.npy')
sink = np.load(data_dir+'/'+'trolls_prob_bg_gen.npy')

src = np.load(data_dir+'/'+'trolls_prob_fg_01.npy')
sink = np.load(data_dir+'/'+'trolls_prob_bg_01.npy')
data = Image.open(data_dir+'/'+'trolls.png')
if test_gray:
    data = data.convert('L')
else:
    data = data.convert('RGBA')

img = np.array(data)

plt.subplot(2, 2, 1)
if test_gray:
    plt.imshow(img, interpolation='nearest', cmap=cmap_gray)
else:
    plt.imshow(img, interpolation='nearest')

img = img.astype(np.double)

t0 = time.clock()
out = graphcut(img, src, sink, affinity=60)
dt = (time.clock() - t0) * 1000
print "completed in %.2f milliseconds" % dt

plt.subplot(2, 2, 2)
plt.imshow(out, interpolation='nearest')

plt.subplot(2, 2, 3)
plt.imshow(src, interpolation='nearest')

plt.subplot(2, 2, 4)
plt.imshow(sink, interpolation='nearest')

plt.show()

