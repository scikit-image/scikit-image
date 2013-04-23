"""
===================
Dynamic Time Warping
===================

This example shows how to match data point in time varying sequences.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from skimage.graph import dtw

stretch = True

vec1 = [71, 73, 75, 80, 80, 80, 78, 76, 75, 73, 71, 71, 71, 73, 75, 76, 76, 68, 76, 76, 75, 73, 71, 70, 70, 69, 68, 68, 72, 74, 78, 79, 80, 80, 78]
vec2 = [69, 69, 73, 75, 79, 80, 79, 78, 76, 73, 72, 71, 70, 70, 69, 69, 69, 71, 73, 75, 76, 76, 76, 76, 76, 75, 73, 71, 70, 70, 71, 73, 75, 80, 80, 80, 78]

t = np.array(vec1, np.double)
r = np.array(vec2, np.double)

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()

for f, case in [(f1, 1), (f2, 2), (f3, 3)]:
    m = np.zeros((len(t), len(r)))

    path, distance = dtw(t, r, case=case, start_anchor_slack=0, end_anchor_slack=0)

    for point in path:
        m[point[0], point[1]] = 1

    if stretch:
        stretch = float(len(t))/len(r)
    else:
        stretch = 1

    ax = f.add_subplot(111, projection='3d')

    ax.plot(np.arange(len(t)), [0]*len(t), t, 'b')
    ax.plot(np.arange(len(t)), [0]*len(t), t, 'b.')
    ax.plot(stretch*np.arange(len(r)), [1]*len(r), r, 'r')
    ax.plot(stretch*np.arange(len(r)), [1]*len(r), r, 'r.')

    for edge in reversed(path):
        ax.plot([edge[0], stretch*edge[1]], [0, 1], [t[edge[0]], r[edge[1]]],  color="gray")

plt.show()
