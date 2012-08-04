"""
===============================================
Local Binary Pattern for texture classification
===============================================

In this example, we will see how to classify textures based on LBP (Local Binary
Pattern). The histogram of the LBP result is a good measure to classify
textures. For simplicity the histogram distributions are then tested against
each other using the Kullback-Leibler-Divergence.
"""

import os
import glob
import numpy as np
import pylab
import scipy.ndimage as nd
import skimage.feature as ft
from skimage.io import imread
from skimage import data


# settings for LBP
METHOD = 'uniform'
P = 16
R = 2


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def match(refs, img):
    best_score = 10
    best_name = None
    lbp = ft.local_binary_pattern(img, P, R, METHOD)
    hist, _ = np.histogram(lbp, normed=True, bins=P + 2, range=(0, P + 2))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, normed=True, bins=P + 2,
                                   range=(0, P + 2))
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


brick = data.load('brick.png')
grass = data.load('grass.png')
wall = data.load('rough-wall.png')

refs = {
    'brick': ft.local_binary_pattern(brick, P, R, METHOD),
    'grass': ft.local_binary_pattern(grass, P, R, METHOD),
    'wall': ft.local_binary_pattern(wall, P, R, METHOD)
}

print match(refs, nd.rotate(brick, angle=30, reshape=False))
print match(refs, nd.rotate(brick, angle=70, reshape=False))
print match(refs, nd.rotate(grass, angle=145, reshape=False))
