from __future__ import division, print_function, absolute_import

import os

import numpy as np
from numpy.testing import assert_equal

import skimage
from skimage import io

from skimage.morphology import compute_thin_image

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# nose test generators:

# 2D images
def test_simple_2d_images():
    for fname in ("strip", "loop", "cross", "two-hole"):
        yield check_skel, fname

# trivial 3D images
def test_simple_3d():
    for fname in ['3/stack', '4/stack']:
        yield check_skel_3d, fname


def get_data_path():
    # XXX this is a bad temp hack
    return os.path.join(os.path.split(skimage.__file__)[0],
                        'morphology',
                        'tests',
                        'data')


def check_skel(fname, viz=False):
    # compute the thin image and compare the result to that of ImageJ

    img = np.loadtxt(os.path.join(get_data_path(), fname+'.txt'), dtype=np.uint8)

    if viz:
        ax = _viz(img, **dict(marker='s', color='b', s=99, alpha=0.2))

    # compute
    img1_2d = compute_thin_image(img)

    if viz:
        ax = _viz(img1_2d, ax, **dict(marker='o', color='r',
                                      s=80, alpha=0.7, label='us'))

    # compare to FIJI
    img_f = np.loadtxt(os.path.join(get_data_path(), fname+'_fiji.txt'), dtype=np.uint8)

    if not viz:
        # actually compare images
        assert_equal(img1_2d, img_f)
    else:
        ax = _viz(img_f, ax, **dict(marker='o', color='g', s=45, label='fiji'))

        ax.legend()
        ax.grid(True)

        def yformatter(val, pos):
            return int(img.shape[1] - val + 1)
        def xformatter(val, pos):
            return int(val + 1)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(xformatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(yformatter))

        plt.show()


def _viz(img, ax=None, **kwds):
    if ax is None:
        import matplotlib.pyplot as plt
        fix, ax = plt.subplots()

    x, y = np.nonzero(img)
    ax.scatter(y, img.shape[1] - x, **kwds)
    return ax


def check_skel_3d(fname):
    img = io.imread(os.path.join(get_data_path(), fname+'.tif'))
    img_f = io.imread(os.path.join(get_data_path(), fname+'_fiji.tif'))

    img_s = compute_thin_image(img)
    assert_equal(img_s, img_f)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("Expect an image name from the data/ directory.")
    check_skel(sys.argv[1], True)
