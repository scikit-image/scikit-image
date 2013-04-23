from __future__ import division

from skimage.graph import dtw
import numpy as np

import time

N = 1000
x = np.random.random(N)
y = np.random.random(N)

iterations = 100

t0 = time.clock()
for i in range(iterations):
    dtw(x, y)
dt = (time.clock() - t0) / iterations * 1000

print "%.2f milliseconds per iteration (mean)" % dt
