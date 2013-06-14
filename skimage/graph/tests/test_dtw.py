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
