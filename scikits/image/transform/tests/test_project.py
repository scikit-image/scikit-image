import numpy as np
from numpy.testing import assert_array_almost_equal

from scikits.image.transform.project import _stackcopy
from scikits.image.transform import homography, fast_homography
from scikits.image import data
from scikits.image.color import rgb2gray

def test_stackcopy():
    layers = 4
    x = np.empty((3, 3, layers))
    y = np.eye(3, 3)
    _stackcopy(x, y)
    for i in range(layers):
        assert_array_almost_equal(x[...,i], y)

def test_homography():
    x = np.arange(9).reshape((3, 3)) + 1
    theta = -np.pi/2
    M = np.array([[np.cos(theta),-np.sin(theta),0],
                  [np.sin(theta), np.cos(theta),2],
                  [0,             0,            1]])
    x90 = homography(x, M, order=1)
    assert_array_almost_equal(x90, np.rot90(x))

def test_fast_homography():
    img = rgb2gray(data.lena()).astype(np.uint8)
    img = img[:, :100]
    
    theta = np.deg2rad(30)
    scale = 0.5
    tx, ty = 50, 50

    H = np.eye(3)
    S = scale * np.sin(theta)
    C = scale * np.cos(theta)

    H[:2, :2] = [[C, -S], [S, C]]
    H[:2, 2] = [tx, ty]

    for mode in ('constant', 'mirror', 'wrap'):
        print 'Transform mode:', mode

        p0 = homography(img, H, mode=mode, order=1)
        p1 = fast_homography(img, H, mode=mode)
        p1 = np.round(p1)

        ## import matplotlib.pyplot as plt
        ## f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
        ## ax0.imshow(img)
        ## ax1.imshow(p0, cmap=plt.cm.gray)
        ## ax2.imshow(p1, cmap=plt.cm.gray)
        ## ax3.imshow(np.abs(p0 - p1), cmap=plt.cm.gray)
        ## plt.show()

        d = np.mean(np.abs(p0 - p1))
        print "delta=", d
        assert d < 0.2
    

if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
