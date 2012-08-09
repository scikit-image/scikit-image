"""
These tests are originally part of CellProfiler, code licensed under both GPL and BSD licenses.

Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky
"""
import numpy as np

from skimage.morphology.greyreconstruct import reconstruction


def test_zeros():
    """Test reconstruction with image and mask of zeros"""
    assert np.all(reconstruction(np.zeros((5, 7)), np.zeros((5, 7))) == 0)


def test_image_equals_mask():
    """Test reconstruction where the image and mask are the same"""
    assert np.all(reconstruction(np.ones((7, 5)), np.ones((7, 5))) == 1)


def test_image_less_than_mask():
    """Test reconstruction where the image is uniform and less than mask"""
    image = np.ones((5, 5))
    mask = np.ones((5, 5)) * 2
    assert np.all(reconstruction(image, mask) == 1)


def test_one_image_peak():
    """Test reconstruction with one peak pixel"""
    image = np.ones((5, 5))
    image[2, 2] = 2
    mask = np.ones((5, 5)) * 3
    assert np.all(reconstruction(image, mask) == 2)


def test_two_image_peaks():
    """Test reconstruction with two peak pixels isolated by the mask"""
    image = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 2, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 3, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1]])

    mask = np.array([[4, 4, 4, 1, 1, 1, 1, 1],
                     [4, 4, 4, 1, 1, 1, 1, 1],
                     [4, 4, 4, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 4, 4, 4],
                     [1, 1, 1, 1, 1, 4, 4, 4],
                     [1, 1, 1, 1, 1, 4, 4, 4]])

    expected = np.array([[2, 2, 2, 1, 1, 1, 1, 1],
                         [2, 2, 2, 1, 1, 1, 1, 1],
                         [2, 2, 2, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 3, 3, 3],
                         [1, 1, 1, 1, 1, 3, 3, 3],
                         [1, 1, 1, 1, 1, 3, 3, 3]])
    assert np.all(reconstruction(image, mask) == expected)


def test_zero_image_one_mask():
    """Test reconstruction with an image of all zeros and a mask that's not"""
    result = reconstruction(np.zeros((10, 10)), np.ones((10, 10)))
    assert np.all(result == 0)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
