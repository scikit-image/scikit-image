from __future__ import division

from skimage.graph import dtw
import numpy as np

def test_identical():
    vec1 = [5*np.sin(i * 0.1 * np.pi) for i in range(0, 50)]
    vec2 = [5*np.sin(i * 0.1 * np.pi) for i in range(0, 50)]

    t = np.array(vec1, np.double)
    r = np.array(vec2, np.double)

    path, distance = dtw(t, r, case=2, start_anchor_slack=0, end_anchor_slack=0)

    for edge in path:
        assert(edge[0] == edge[1])

def dist_l2_norm(a, b):
    return np.sqrt((a-b)**2)

def test_user_distance():
#    vec1 = [5*np.sin(i * 0.1 * np.pi) for i in range(0, 50)]
#    vec2 = [5*np.sin(i * 0.1 * np.pi) for i in range(0, 50)]

    vec1 = [71, 73, 75, 80, 80, 80, 78, 76, 75, 73, 71, 71, 71, 73, 75, 76,
           76, 68, 76, 76, 75, 73, 71, 70, 70, 69, 68, 68, 72, 74, 78, 79,
           80,  80, 78]
    vec2 = [69, 69, 73, 75, 79, 80, 79, 78, 76, 73, 72, 71, 70, 70, 69, 69,
           69, 71, 73, 75, 76, 76, 76, 76, 76, 75, 73, 71, 70, 70, 71, 73,
           75, 80, 80, 80, 78]

    x = np.array(vec1, np.double)
    y = np.array(vec2, np.double)

    m = len(x)
    n = len(y)

    distance = np.zeros((m + 2, n + 2)) + np.finfo(np.double).max
    distance[1, 1] = 0

    for i in range(2, 3):
        distance[i, 2] = dist_l2_norm(x[i - 2], y[0])

    for j in range(2, 3):
        distance[2, j] = dist_l2_norm(x[0], y[j - 2])

    # Populate distance matrix
    for i in range(3, m + 2):
        for j in range(3, n + 2):
            c = min(distance[i - 1, j - 1],
                       distance[i - 1, j],
                       distance[i, j - 1]
                       )

            distance[i, j] = dist_l2_norm(x[i - 2], y[j - 2]) + c

    path, distance = dtw(x, y, case=2, start_anchor_slack=0,
        end_anchor_slack=0, distance=distance)

    path2, distance2 = dtw(x, y, case=2, start_anchor_slack=0,
        end_anchor_slack=0)

    i = 0
    for i in range(len(path)):
        assert(path[i] == path2[i])

test_identical()
test_user_distance()