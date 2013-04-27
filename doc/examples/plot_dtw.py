"""
===================
Dynamic Time Warping
===================

This example shows how to match data point in time varying sequences.

"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.graph import dtw

stretch = True
v_offset = 100
axis_padding = 0.05

vec1 = [71, 73, 75, 80, 80, 80, 78, 76, 75, 73, 71, 71, 71, 73, 75, 76, 76, 68,
        76, 76, 75, 73, 71, 70, 70, 69, 68, 68, 72, 74, 78, 79, 80, 80, 78]
vec2 = [69, 69, 73, 75, 79, 80, 79, 78, 76, 73, 72, 71, 70, 70, 69, 69, 69, 71,
        73, 75, 76, 76, 76, 76, 76, 75, 73, 71, 70, 70, 71, 73, 75, 80, 80, 80,
        78]

t = np.array(vec1, np.double)
r = np.array(vec2, np.double)

f = plt.figure()

m = np.zeros((len(t), len(r)))

path, distance = dtw(t, r)

for point in path:
    m[point[0], point[1]] = 1

if stretch:
    stretch = float(len(t)) / len(r)
else:
    stretch = 1

ax = f.add_subplot(111)

ax.plot(np.arange(len(t)), t+v_offset, 'b')
ax.plot(np.arange(len(t)), t+v_offset, 'b.')
ax.plot(stretch * np.arange(len(r)), r, 'r')
ax.plot(stretch * np.arange(len(r)), r, 'r.')

for edge in reversed(path):
    ax.plot([edge[0], stretch * edge[1]], [v_offset+t[edge[0]], r[edge[1]]],
        color="gray")

ax.autoscale_view(True,True,True)

x1, x2, y1, y2 = plt.axis()
x_pad = axis_padding*(x2-x1)
y_pad = axis_padding*(y2-y1)

plt.axis((x1-x_pad, x2+x_pad, y1-y_pad, y2+y_pad))

plt.show()
