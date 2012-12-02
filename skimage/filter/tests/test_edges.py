import numpy as np
from numpy.testing import assert_array_almost_equal as assert_close

import skimage.filter as F


def test_sobel_zeros():
    """Sobel on an array of all zeros"""
    result = F.sobel(np.zeros((10, 10)), np.ones((10, 10), bool))
    assert (np.all(result == 0))

def test_sobel_mask():
    """Sobel on a masked array should be zero"""
    np.random.seed(0)
    result = F.sobel(np.random.uniform(size=(10, 10)),
                     np.zeros((10, 10), bool))
    assert (np.all(result == 0))

def test_sobel_horizontal():
    """Sobel on an edge should be a horizontal line"""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = F.sobel(image)
    # Fudge the eroded points
    i[np.abs(j) == 5] = 10000
    assert (np.all(result[i == 0] == 1))
    assert (np.all(result[np.abs(i) > 1] == 0))

def test_sobel_vertical():
    """Sobel on a vertical edge should be a vertical line"""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = F.sobel(image)
    j[np.abs(i) == 5] = 10000
    assert (np.all(result[j == 0] == 1))
    assert (np.all(result[np.abs(j) > 1] == 0))


def test_hsobel_zeros():
    """Horizontal sobel on an array of all zeros"""
    result = F.hsobel(np.zeros((10, 10)), np.ones((10, 10), bool))
    assert (np.all(result == 0))

def test_hsobel_mask():
    """Horizontal Sobel on a masked array should be zero"""
    np.random.seed(0)
    result = F.hsobel(np.random.uniform(size=(10, 10)),
                      np.zeros((10, 10), bool))
    assert (np.all(result == 0))

def test_hsobel_horizontal():
    """Horizontal Sobel on an edge should be a horizontal line"""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = F.hsobel(image)
    # Fudge the eroded points
    i[np.abs(j) == 5] = 10000
    assert (np.all(result[i == 0] == 1))
    assert (np.all(result[np.abs(i) > 1] == 0))

def test_hsobel_vertical():
    """Horizontal Sobel on a vertical edge should be zero"""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = F.hsobel(image)
    assert (np.all(result == 0))


def test_vsobel_zeros():
    """Vertical sobel on an array of all zeros"""
    result = F.vsobel(np.zeros((10, 10)), np.ones((10, 10), bool))
    assert (np.all(result == 0))

def test_vsobel_mask():
    """Vertical Sobel on a masked array should be zero"""
    np.random.seed(0)
    result = F.vsobel(np.random.uniform(size=(10, 10)),
                      np.zeros((10, 10), bool))
    assert (np.all(result == 0))

def test_vsobel_vertical():
    """Vertical Sobel on an edge should be a vertical line"""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = F.vsobel(image)
    # Fudge the eroded points
    j[np.abs(i) == 5] = 10000
    assert (np.all(result[j == 0] == 1))
    assert (np.all(result[np.abs(j) > 1] == 0))

def test_vsobel_horizontal():
    """vertical Sobel on a horizontal edge should be zero"""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = F.vsobel(image)
    eps = .000001
    assert (np.all(np.abs(result) < eps))


def test_prewitt_zeros():
    """Prewitt on an array of all zeros"""
    result = F.prewitt(np.zeros((10, 10)), np.ones((10, 10), bool))
    assert (np.all(result == 0))

def test_prewitt_mask():
    """Prewitt on a masked array should be zero"""
    np.random.seed(0)
    result = F.prewitt(np.random.uniform(size=(10, 10)),
                       np.zeros((10, 10), bool))
    eps = .000001
    assert (np.all(np.abs(result) < eps))

def test_prewitt_horizontal():
    """Prewitt on an edge should be a horizontal line"""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = F.prewitt(image)
    # Fudge the eroded points
    i[np.abs(j) == 5] = 10000
    eps = .000001
    assert (np.all(result[i == 0] == 1))
    assert (np.all(np.abs(result[np.abs(i) > 1]) < eps))

def test_prewitt_vertical():
    """Prewitt on a vertical edge should be a vertical line"""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = F.prewitt(image)
    eps = .000001
    j[np.abs(i) == 5] = 10000
    assert (np.all(result[j == 0] == 1))
    assert (np.all(np.abs(result[np.abs(j) > 1]) < eps))


def test_hprewitt_zeros():
    """Horizontal prewitt on an array of all zeros"""
    result = F.hprewitt(np.zeros((10, 10)), np.ones((10, 10), bool))
    assert (np.all(result == 0))

def test_hprewitt_mask():
    """Horizontal prewitt on a masked array should be zero"""
    np.random.seed(0)
    result = F.hprewitt(np.random.uniform(size=(10, 10)),
                        np.zeros((10, 10), bool))
    eps = .000001
    assert (np.all(np.abs(result) < eps))

def test_hprewitt_horizontal():
    """Horizontal prewitt on an edge should be a horizontal line"""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = F.hprewitt(image)
    # Fudge the eroded points
    i[np.abs(j) == 5] = 10000
    eps = .000001
    assert (np.all(result[i == 0] == 1))
    assert (np.all(np.abs(result[np.abs(i) > 1]) < eps))

def test_hprewitt_vertical():
    """Horizontal prewitt on a vertical edge should be zero"""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = F.hprewitt(image)
    eps = .000001
    assert (np.all(np.abs(result) < eps))


def test_vprewitt_zeros():
    """Vertical prewitt on an array of all zeros"""
    result = F.vprewitt(np.zeros((10, 10)), np.ones((10, 10), bool))
    assert (np.all(result == 0))

def test_vprewitt_mask():
    """Vertical prewitt on a masked array should be zero"""
    np.random.seed(0)
    result = F.vprewitt(np.random.uniform(size=(10, 10)),
                        np.zeros((10, 10), bool))
    assert (np.all(result == 0))

def test_vprewitt_vertical():
    """Vertical prewitt on an edge should be a vertical line"""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = F.vprewitt(image)
    # Fudge the eroded points
    j[np.abs(i) == 5] = 10000
    assert (np.all(result[j == 0] == 1))
    eps = .000001
    assert (np.all(np.abs(result[np.abs(j) > 1]) < eps))

def test_vprewitt_horizontal():
    """Vertical prewitt on a horizontal edge should be zero"""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = F.vprewitt(image)
    eps = .000001
    assert (np.all(np.abs(result) < eps))


def test_horizontal_mask_line():
    """Horizontal edge filters mask pixels surrounding input mask."""
    vgrad, _ = np.mgrid[:1:11j, :1:11j] # vertical gradient with spacing 0.1
    vgrad[5, :] = 1                     # bad horizontal line

    mask = np.ones_like(vgrad)
    mask[5, :] = 0                      # mask bad line

    expected = np.zeros_like(vgrad)
    expected[1:-1, 1:-1] = 0.2          # constant gradient for most of image,
    expected[4:7, 1:-1] = 0             # but line and neighbors masked

    for grad_func in (F.hprewitt, F.hsobel):
        result = grad_func(vgrad, mask)
        yield assert_close, result, expected


def test_vertical_mask_line():
    """Vertical edge filters mask pixels surrounding input mask."""
    _, hgrad = np.mgrid[:1:11j, :1:11j] # horizontal gradient with spacing 0.1
    hgrad[:, 5] = 1                     # bad vertical line

    mask = np.ones_like(hgrad)
    mask[:, 5] = 0                      # mask bad line

    expected = np.zeros_like(hgrad)
    expected[1:-1, 1:-1] = 0.2          # constant gradient for most of image,
    expected[1:-1, 4:7] = 0             # but line and neighbors masked

    for grad_func in (F.vprewitt, F.vsobel):
        result = grad_func(hgrad, mask)
        yield assert_close, result, expected


if __name__ == "__main__":
    from numpy import testing
    testing.run_module_suite()
