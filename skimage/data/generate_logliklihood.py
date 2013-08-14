from sklearn import mixture
import numpy as np
import Image
from skimage import data_dir

data = Image.open(data_dir+'/'+'trolls_small.png')
if data.mode != 'RGB':
    data = data.convert('RGB')

img = np.array(data).astype(np.float)

gmm = mixture.GMM(4)

#s = slice(160, 175)
s = slice(25, 50)
area = [s, s]

samples = img[area].reshape(-1, 3)
gmm.fit(samples)

out = gmm.score(img.reshape(-1, 3)).reshape((img.shape[0], img.shape[1]))
out[area] = 0

np.save(data_dir+'/'+'trolls_bg', out)

import matplotlib.pyplot as plt
plt.imshow(out, interpolation='nearest')
plt.show()

True