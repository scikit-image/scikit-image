import numpy as np
import scipy

from skimage.feature import hog


def test_histogram_of_oriented_gradients():
    # Replace with skimage.data.lena() after merge
    img = scipy.misc.lena()[:256, :].astype(np.int8)

    fd = hog(img, orientations=9, pixels_per_cell=(8, 8),
             cells_per_block=(1, 1))

    assert len(fd) == 9 * (256 // 8) * (512 // 8)

if __name__ == '__main__':
    from numpy.testing import run_module_suite
    run_module_suite()
