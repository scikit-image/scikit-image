from sklearn import mixture
import numpy as np
import Image
from skimage import data_dir
import matplotlib.pyplot as plt

jet = plt.get_cmap('jet')
jet.set_under(color='k')
jet.set_over(color='w')

_img = Image.open(data_dir+'/'+'trolls.png')
if _img.mode != 'RGB':
    _img = _img.convert('RGB')

img = np.array(_img)
img = img.astype(np.double)

samples_bg = []
samples_fg = []

areas_bg = [(slice(425, 450), slice(50, 75))]
areas_fg = [(slice(300, 330), slice(330, 350))]

for area in areas_bg:
    samples_bg = img[area].reshape(-1, 3)

for area in areas_fg:
    samples_fg = img[area].reshape(-1, 3)

#points_fg = np.load(data_dir+'/'+'trolls_prob_fg_01.npy')
#points_bg = np.load(data_dir+'/'+'trolls_prob_bg_01.npy')
#points_fg = points_fg.swapaxes(1, 0)
#points_bg = points_bg.swapaxes(1, 0)
#
#samples_fg = []
#for point in points_fg:
#    samples_fg.append(img[tuple(point)])
#
#for point in points_bg:
#    samples_bg.append(img[tuple(point)])

gmm_fg = mixture.GMM(4)
gmm_bg = mixture.GMM(8)

gmm_bg.fit(samples_bg)
gmm_fg.fit(samples_fg)

sink = -gmm_bg.score(img.reshape(-1, 3)).reshape((img.shape[0], img.shape[1]))
src = -gmm_fg.score(img.reshape(-1, 3)).reshape((img.shape[0], img.shape[1]))

np.save(data_dir+'/'+'trolls_prob_bg_gen.npy', sink)
np.save(data_dir+'/'+'trolls_prob_fg_gen.npy', src)

src_min = np.exp(-src).min()
src_max = np.exp(-src).max()

sink_min = np.exp(-sink).min()
sink_max = np.exp(-sink).max()

#for area in areas_bg:
#    src[area] = -np.log(0)
#    sink[area] = -np.log(1)
#
#for area in areas_fg:
#    src[area] = -np.log(1)
#    sink[area] = -np.log(0)

#for point in points_fg:
#    src[tuple(point)] = -np.log(0)
#    sink[tuple(point)] = -np.log(1)
#
#for point in points_bg:
#    src[tuple(point)] = -np.log(1)
#    sink[tuple(point)] = -np.log(0)

plt.subplot(5, 2, 1)
plt.imshow(img/255, interpolation='nearest')

plt.subplot(5, 2, 2)
plt.imshow(src-sink, interpolation='nearest', cmap=jet, vmin=0)

plt.subplot(5, 2, 3)
plt.imshow(src, interpolation='nearest', cmap=jet)

plt.subplot(5, 2, 4)
plt.imshow(sink, interpolation='nearest', cmap=jet)

plt.subplot(5, 2, 5)
plt.imshow(np.exp(-src), interpolation='nearest', cmap=jet, vmin=src_min, vmax=src_max)

plt.subplot(5, 2, 6)
plt.imshow(np.exp(-sink), interpolation='nearest', cmap=jet, vmin=sink_min, vmax=sink_max)

d0 = int(np.sqrt(len(samples_fg)))
show_samples_fg = np.array(samples_fg[0:d0*d0]).reshape(d0, d0, 3)

plt.subplot(5, 2, 7)
plt.imshow(show_samples_fg/255, interpolation='nearest')

d0 = int(np.sqrt(len(samples_bg)))
show_samples_bg = np.array(samples_bg[0:d0*d0]).reshape(d0, d0, 3)

plt.subplot(5, 2, 8)
plt.imshow(show_samples_bg/255, interpolation='nearest')

plt.subplot(5, 2, 9)
plt.imshow(gmm_fg.means_.reshape(1, 4, 3)/255, interpolation='nearest')

plt.subplot(5, 2, 10)
plt.imshow(gmm_bg.means_.reshape(1, 8, 3)/255, interpolation='nearest')


plt.show()

True