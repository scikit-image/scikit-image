
import numpy as np
from skimage.segmentation import morph_acwe, morph_gac
from skimage.segmentation import morphsnakes
from numpy.testing import assert_array_equal
import pytest


def circle_levelset(shape, center, radius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[[slice(i) for i in shape]].T - center
    phi = radius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.int8(phi > 0)
    return u


def gaussian_blob():

    coords = np.mgrid[-5:6, -5:6]
    sqrdistances = (coords ** 2).sum(0)
    return np.exp(-sqrdistances / 10)


def test_morphsnakes_incorrect_image_shape():
    img = np.zeros((10, 10, 3))
    ls = np.zeros((10, 9))

    with pytest.raises(ValueError):
        morph_acwe(img, init_level_set=ls, iterations=1)
    with pytest.raises(ValueError):
        morph_gac(img, init_level_set=ls, iterations=1)


def test_morphsnakes_incorrect_ndim():

    img = np.zeros((4, 4, 4, 4))
    ls = np.zeros((4, 4, 4, 4))

    with pytest.raises(ValueError):
        morph_acwe(img, init_level_set=ls, iterations=1)
    with pytest.raises(ValueError):
        morph_gac(img, init_level_set=ls, iterations=1)


def test_morphsnakes_black():

    img = np.zeros((11, 11))
    ls = circle_levelset(img.shape, (5, 5), 3)

    ref_zeros = np.zeros(img.shape, dtype=np.int8)
    ref_ones = np.ones(img.shape, dtype=np.int8)

    acwe_ls = morph_acwe(img, ls, iterations=6)
    assert_array_equal(acwe_ls, ref_zeros)

    gac_ls = morph_gac(img, ls, iterations=6)
    assert_array_equal(gac_ls, ref_zeros)

    gac_ls2 = morph_gac(img, ls, iterations=6, balloon=1, threshold=-1,
                        smoothing=0)
    assert_array_equal(gac_ls2, ref_ones)

    assert acwe_ls.dtype == gac_ls.dtype == gac_ls2.dtype == np.int8


def test_morphsnakes_simple_shape_acwe():

    img = gaussian_blob()
    ls1 = circle_levelset(img.shape, (5, 5), 3)
    ls2 = circle_levelset(img.shape, (5, 5), 6)

    acwe_ls1 = morph_acwe(img, ls1, iterations=10)
    acwe_ls2 = morph_acwe(img, ls2, iterations=10)

    assert_array_equal(acwe_ls1, acwe_ls2)

    assert acwe_ls1.dtype == acwe_ls2.dtype == np.int8


def test_morphsnakes_simple_shape_gac():

    img = np.float_(circle_levelset((11, 11), (5, 5), 3.5))
    gimg = morphsnakes.gborders(img, alpha=10.0)
    ls = circle_levelset(img.shape, (5, 5), 6)

    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                   dtype=np.int8)

    gac_ls = morph_gac(gimg, ls, iterations=10, balloon=-1)

    assert_array_equal(gac_ls, ref)

    assert gac_ls.dtype == np.int8


if __name__ == "__main__":
    np.testing.run_module_suite()
