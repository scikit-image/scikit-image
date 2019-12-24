import time
import numpy as np
import skimage.filters
import skimage._shared.gpu

image = np.random.random((4000, 4000))
image[:200, :200] += 1
image[300:, 300] += 0.5
currtime = time.time()

output = skimage.filters.gabor(image,0.1)
print(output)
delta = time.time() - currtime
print('taken',delta)