# Author: Damian Eads

import os.path

import numpy as np
from numpy.testing import *

from skimage import data_dir
from skimage.io import *
from skimage import data_dir
from skimage.morphology import selem


class TestSElem():

    def test_square_selem(self):
        for k in range(0, 5):
            actual_mask = selem.square(k)
            expected_mask = np.ones((k, k), dtype='uint8')
            assert_equal(expected_mask, actual_mask)

    def test_rectangle_selem(self):
        for i in range(0, 5):
            for j in range(0, 5):
                actual_mask = selem.rectangle(i, j)
                expected_mask = np.ones((i, j), dtype='uint8')
                assert_equal(expected_mask, actual_mask)

    def strel_worker(self, fn, func):
        matlab_masks = np.load(os.path.join(data_dir, fn))
        k = 0
        for arrname in sorted(matlab_masks):
            expected_mask = matlab_masks[arrname]
            actual_mask = func(k)
            if (expected_mask.shape == (1,)):
                expected_mask = expected_mask[:, np.newaxis]
            assert_equal(expected_mask, actual_mask)
            k = k + 1

    def strel_worker_3d(self, fn, func):
        matlab_masks = np.load(os.path.join(data_dir, fn))
        k = 0
        for arrname in sorted(matlab_masks):
            expected_mask = matlab_masks[arrname]
            actual_mask = func(k)
            if (expected_mask.shape == (1,)):
                expected_mask = expected_mask[:, np.newaxis]
            # Test center slice for each dimension. This gives a good
            # indication of validity without the need for a 3D reference
            # mask.
            c = int(expected_mask.shape[0]/2)
            assert_equal(expected_mask, actual_mask[c,:,:])
            assert_equal(expected_mask, actual_mask[:,c,:])
            assert_equal(expected_mask, actual_mask[:,:,c])
            k = k + 1

    def test_selem_disk(self):
        self.strel_worker("disk-matlab-output.npz", selem.disk)

    def test_selem_diamond(self):
        self.strel_worker("diamond-matlab-output.npz", selem.diamond)

    def test_selem_ball(self):
        self.strel_worker_3d("disk-matlab-output.npz", selem.ball)

    def test_selem_octahedron(self):
        self.strel_worker_3d("diamond-matlab-output.npz", selem.octahedron)

    def test_selem_octagon(self):
        expected_mask1 = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=np.uint8)
        actual_mask1 = selem.octagon(5, 3)
        expected_mask2 = np.array([[0, 1, 0],
                                   [1, 1, 1],
                                   [0, 1, 0]], dtype=np.uint8)
        actual_mask2 = selem.octagon(1, 1)
        assert_equal(expected_mask1, actual_mask1)
        assert_equal(expected_mask2, actual_mask2)

    def test_selem_star(self):
        expected_mask1 = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        actual_mask1 = selem.star(4)
        expected_mask2 = np.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]], dtype=np.uint8)
        actual_mask2 = selem.star(1)
        assert_equal(expected_mask1, actual_mask1)
        assert_equal(expected_mask2, actual_mask2)


if __name__ == '__main__':
    np.testing.run_module_suite()
