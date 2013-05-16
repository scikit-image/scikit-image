"""
===================
Dynamic Time Warping
===================
DTW is an algorithm for measuring similarity between two sequences which
may vary in time or speed

This example shows how to match data point in time varying sequences.

"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.graph import dtw

stretch = True
v_offset = 10
axis_padding = 0.05

vec1 = [71, 73, 75, 80, 80, 80, 78, 76, 75, 73, 71, 71, 71, 73, 75, 76, 76, 68,
        76, 76, 75, 73, 71, 70, 70, 69, 68, 68, 72, 74, 78, 79, 80, 80, 78]
vec2 = [69, 69, 73, 75, 79, 80, 79, 78, 76, 73, 72, 71, 70, 70, 69, 69, 69, 71,
        73, 75, 76, 76, 76, 76, 76, 75, 73, 71, 70, 70, 71, 73, 75, 80, 80, 80,
        78]

t = np.array(vec1, np.double)
r = np.array(vec2, np.double)

m = np.zeros((len(t), len(r)))

path, distance = dtw(t, r, case=2)

for point in path:
    m[point[0], point[1]] = 1

if stretch:
    stretch = float(len(t)) / len(r)
else:
    stretch = 1

f, ax = plt.subplots()

ax.plot(np.arange(len(t)), t+v_offset, 'b.-')
ax.plot(stretch * np.arange(len(r)), r, 'r.-')

for edge in reversed(path):
    ax.plot([edge[0], stretch * edge[1]], [v_offset+t[edge[0]], r[edge[1]]],
        color="gray")

plt.margins(0.1)

plt.show()
