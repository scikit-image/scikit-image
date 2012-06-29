import numpy as np
from numpy.testing import assert_array_equal, run_module_suite

from skimage.morphology import label


class TestConnectedComponents:
    def setup(self):
        self.x = np.array([[0, 0, 3, 2, 1, 9],
                           [0, 1, 1, 9, 2, 9],
                           [0, 0, 1, 9, 9, 9],
                           [3, 1, 1, 5, 3, 0]])

        self.labels = np.array([[0, 0, 1, 2, 3, 4],
                                [0, 5, 5, 4, 2, 4],
                                [0, 0, 5, 4, 4, 4],
                                [6, 5, 5, 7, 8, 9]])

    def test_basic(self):
        assert_array_equal(label(self.x), self.labels)

        # Make sure data wasn't modified
        assert self.x[0, 2] == 3

    def test_random(self):
        x = (np.random.random((20, 30)) * 5).astype(np.int)

        labels = label(x)
        n = labels.max()
        for i in range(n):
            values = x[labels == i]
            assert np.all(values == values[0])

    def test_diag(self):
        x = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]])
        assert_array_equal(label(x),
                           x)

    def test_4_vs_8(self):
        x = np.array([[0, 1],
                      [1, 0]], dtype=int)
        assert_array_equal(label(x, 4),
                           [[0, 1],
                            [2, 3]])
        assert_array_equal(label(x, 8),
                           [[0, 1],
                            [1, 0]])

    def test_background(self):
        x = np.array([[1, 0, 0],
                      [1, 1, 5],
                      [0, 0, 0]])

        assert_array_equal(label(x), [[0, 1, 1],
                                      [0, 0, 2],
                                      [3, 3, 3]])

        assert_array_equal(label(x, background=0),
                           [[0, -1, -1],
                            [0,  0,  1],
                            [-1, -1, -1]])

    def test_background_two_regions(self):
        x = np.array([[0, 0, 6],
                      [0, 0, 6],
                      [5, 5, 5]])

        assert_array_equal(label(x, background=0),
                           [[-1, -1, 0],
                            [-1, -1, 0],
                            [ 1,  1, 1]])

    def test_background_one_region_center(self):
        x = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]])

        assert_array_equal(label(x, neighbors=4, background=0),
                           [[-1, -1, -1],
                            [-1,  0, -1],
                            [-1, -1, -1]])


if __name__ == "__main__":
    run_module_suite()
