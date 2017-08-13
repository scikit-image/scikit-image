
import numpy as np
from skimage.segmentation import morph_acwe, morph_gac
# from numpy.testing import assert_array_equal
import pytest

def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u

def test_morphsnakes_incorrect_image_shape():
    img = np.zeros((10, 10, 3))
    ls = np.zeros((10, 9))
    with pytest.raises(ValueError):
        morph_acwe(img, init_level_set=ls, iterations=1)
    with pytest.raises(ValueError):
        morph_gac(img, init_level_set=ls, iterations=1)

if __name__ == "__main__":
    np.testing.run_module_suite()
