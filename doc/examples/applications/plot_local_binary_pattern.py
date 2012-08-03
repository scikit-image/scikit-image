"""
===============================================
Local Binary Pattern for texture classification
===============================================

In this example, we will see how to classify textures based on LBP (Local
Binary Pattern). This example uses different rotated textures, taken from
http://sipi.usc.edu/database/database.php?volume=rotate.

The histogram of the LBP result is a good measure to classify textures. For
simplicity the histogram distributions are then tested against each other using
the Kullback-Leibler-Divergence.

Preparation
===========

First you need to download and extract the texture image set from
http://sipi.usc.edu/database/database.php?volume=rotate. Make sure you change
the path to the extracted images in the script (`IMAGE_FOLDER`). You must run
the `dump_refs` function only once, so the computation is faster. Finally you
can match any of the rotated images against the reference textures.
"""

import os
import glob
import numpy as np
import pylab
import skimage.feature as ft
from skimage.io import imread


IMAGE_FOLDER = 'images'
REF_DUMP_FOLDER = 'refs'
METHOD = 'uniform'
P = 16
R = 2


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def dump_refs(refs):
    os.mkdir(REF_DUMP_FOLDER)
    file_names = glob.glob('%s/*.000.tiff' % IMAGE_FOLDER)
    for file_name in file_names:
        name, _ = os.path.splitext(os.path.basename(file_name))
        lbp = ft.local_binary_pattern(ref, P, R, METHOD)
        np.save('refs/%s.npy' % name, lbp)


def load_refs():
    file_names = glob.glob('refs/*.000.npy')
    refs = {}
    for file_name in file_names:
        name, _ = os.path.splitext(os.path.basename(file_name))
        refs[name] = np.load(file_name)
    return refs


def match(refs, img):
    best_score = 10
    best_name = None
    lbp = ft.local_binary_pattern(img, P, R, METHOD)
    hist, _ = np.histogram(lbp, normed=True, bins=P+2, range=(0, P+2))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, normed=True, bins=P+2, range=(0, P+2))
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


# compute LBP for each reference image once and dump result, once run this
# should be commented out
dump_refs(refs)

# match any rotated image against reference textures
refs = load_refs()
img = imread(os.path.join(IMAGE_FOLDER, 'grass.060.tiff'))
print match(refs, img) # grass.000
img = imread(os.path.join(IMAGE_FOLDER, 'brick.060.tiff'))
print match(refs, img) # brick.000
img = imread(os.path.join(IMAGE_FOLDER, 'bubbles.060.tiff'))
print match(refs, img) # bubbles.000
