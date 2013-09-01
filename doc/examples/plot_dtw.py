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

def dist_l2_norm(a, b):
    return np.sqrt((a-b)**2)

stretch = True
v_offset = 10
axis_padding = 0.05

vec1 = [71, 73, 75, 80, 80, 80, 78, 76, 75, 73, 71, 71, 71, 73, 75, 76, 76, 68,
        76, 76, 75, 73, 71, 70, 70, 69, 68, 68, 72, 74, 78, 79, 80, 80, 78]
vec2 = [69, 69, 73, 75, 79, 80, 79, 78, 76, 73, 72, 71, 70, 70, 69, 69, 69, 71,
        73, 75, 76, 76, 76, 76, 76, 75, 73, 71, 70, 70, 71, 73, 75, 80, 80, 80,
        78]

vec1 = [0,0,0,0,1,1,2,2,3,2,1,1,0,0,0,0]
vec2 = [0,0,1,1,2,2,3,3,3,3,2,2,1,1,0,0]

#vec1 = [5*np.sin(i * 0.1 * np.pi) for i in range(0, 50)]
#vec2 = [5*np.sin(i * 0.1 * np.pi) for i in range(0, 50)]

t = np.array(vec1, np.double)
r = np.array(vec2, np.double)

m = len(t)
n = len(r)

distance = np.zeros((m + 2, n + 2)) + np.finfo(np.double).max
distance[1, 1] = 0

for i in range(2, 3):
    distance[i, 2] = dist_l2_norm(t[i - 2], r[0])

for j in range(2, 3):
    distance[2, j] = dist_l2_norm(t[0], r[j - 2])

# Populate distance matrix
for i in range(3, m + 2):
    for j in range(3, n + 2):
        c = min(distance[i - 1, j - 1],
            distance[i - 1, j],
            distance[i, j - 1]
        )

        distance[i, j] = dist_l2_norm(t[i - 2], r[j - 2]) + c

path, distance = dtw(t, r, case=2, start_anchor_slack=0,
    end_anchor_slack=0, distance=distance)

#path, distance = dtw(t, r, case=2, start_anchor_slack=0, end_anchor_slack=0)

m = np.zeros((len(t), len(r)))
for point in path:
    m[point[0], point[1]] = 1

if stretch:
    stretch = float(len(t)) / len(r)
else:
    stretch = 1

f, ax = plt.subplots()

ax.plot(np.arange(len(t)), t+v_offset, 'b.-')
ax.plot(stretch*np.arange(len(r)), r, 'r.-')

for edge in reversed(path):
    ax.plot([edge[0], stretch * edge[1]], [v_offset+t[edge[0]], r[edge[1]]],
        color="gray")

plt.margins(0.1)

plt.show()
