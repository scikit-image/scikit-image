# Author: Damian Eads

import os.path

import numpy as np
from numpy.testing import *

from skimage import data_dir
from skimage.io import *
from skimage import data_dir
from skimage.morphology import *


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

    def test_selem_disk(self):
        self.strel_worker("disk-matlab-output.npz", selem.disk)

    def test_selem_diamond(self):
        self.strel_worker("diamond-matlab-output.npz", selem.diamond)
