import numpy as np
from numpy.testing import assert_array_equal, run_module_suite

from skimage.measure import label
import skimage.measure._ccomp as ccomp
from skimage._shared._warnings import expected_warnings


# The background label value
# is supposed to be changed to 0 soon
BG = -1


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
        with expected_warnings(['`background`']):
            assert_array_equal(label(self.x), self.labels)

        # Make sure data wasn't modified
        assert self.x[0, 2] == 3

    def test_random(self):
        x = (np.random.rand(20, 30) * 5).astype(np.int)

        with expected_warnings(['`background`']):
            labels = label(x)

        n = labels.max()
        for i in range(n):
            values = x[labels == i]
            assert np.all(values == values[0])

    def test_diag(self):
        x = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]])
        with expected_warnings(['`background`']):
            assert_array_equal(label(x), x)

    def test_4_vs_8(self):
        x = np.array([[0, 1],
                      [1, 0]], dtype=int)
        with expected_warnings(['`background`']):
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

        with expected_warnings(['`background`']):
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

        res = label(x, background=0)
        assert_array_equal(res,
                           [[-1, -1, 0],
                            [-1, -1, 0],
                            [+1,  1, 1]])

    def test_background_one_region_center(self):
        x = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]])

        assert_array_equal(label(x, neighbors=4, background=0),
                           [[-1, -1, -1],
                            [-1,  0, -1],
                            [-1, -1, -1]])

    def test_return_num(self):
        x = np.array([[1, 0, 6],
                      [0, 0, 6],
                      [5, 5, 5]])

        with expected_warnings(['`background`']):
            assert_array_equal(label(x, return_num=True)[1], 4)

        assert_array_equal(label(x, background=0, return_num=True)[1], 3)


class TestConnectedComponents3d:
    def setup(self):
        self.x = np.zeros((3, 4, 5), int)
        self.x[0] = np.array([[0, 3, 2, 1, 9],
                              [0, 1, 9, 2, 9],
                              [0, 1, 9, 9, 9],
                              [3, 1, 5, 3, 0]])

        self.x[1] = np.array([[3, 3, 2, 1, 9],
                              [0, 3, 9, 2, 1],
                              [0, 3, 3, 1, 1],
                              [3, 1, 3, 3, 0]])

        self.x[2] = np.array([[3, 3, 8, 8, 0],
                              [2, 3, 9, 8, 8],
                              [2, 3, 0, 8, 0],
                              [2, 1, 0, 0, 0]])

        self.labels = np.zeros((3, 4, 5), int)

        self.labels[0] = np.array([[0, 1, 2, 3, 4],
                                   [0, 5, 4, 2, 4],
                                   [0, 5, 4, 4, 4],
                                   [1, 5, 6, 1, 7]])

        self.labels[1] = np.array([[1, 1, 2, 3, 4],
                                   [0, 1, 4, 2, 3],
                                   [0, 1, 1, 3, 3],
                                   [1, 5, 1, 1, 7]])

        self.labels[2] = np.array([[1,  1, 8, 8, 9],
                                   [10, 1, 4, 8, 8],
                                   [10, 1, 7, 8, 7],
                                   [10, 5, 7, 7, 7]])

    def test_basic(self):
        with expected_warnings(['`background`']):
            labels = label(self.x)
        assert_array_equal(labels, self.labels)

        assert self.x[0, 0, 2] == 2, \
            "Data was modified!"

    def test_random(self):
        x = (np.random.rand(20, 30) * 5).astype(np.int)

        with expected_warnings(['`background`']):
            labels = label(x)

        n = labels.max()
        for i in range(n):
            values = x[labels == i]
            assert np.all(values == values[0])

    def test_diag(self):
        x = np.zeros((3, 3, 3), int)
        x[0, 2, 2] = 1
        x[1, 1, 1] = 1
        x[2, 0, 0] = 1
        with expected_warnings(['`background`']):
            assert_array_equal(label(x), x)

    def test_4_vs_8(self):
        x = np.zeros((2, 2, 2), int)
        x[0, 1, 1] = 1
        x[1, 0, 0] = 1
        label4 = x.copy()
        label4[1, 0, 0] = 2
        with expected_warnings(['`background`']):
            assert_array_equal(label(x, 4), label4)
            assert_array_equal(label(x, 8), x)

    def test_background(self):
        x = np.zeros((2, 3, 3), int)
        x[0] = np.array([[1, 0, 0],
                         [1, 0, 0],
                         [0, 0, 0]])
        x[1] = np.array([[0, 0, 0],
                         [0, 1, 5],
                         [0, 0, 0]])

        lnb = x.copy()
        lnb[0] = np.array([[0, 1, 1],
                           [0, 1, 1],
                           [1, 1, 1]])
        lnb[1] = np.array([[1, 1, 1],
                           [1, 0, 2],
                           [1, 1, 1]])
        lb = x.copy()
        lb[0] = np.array([[0,  BG, BG],
                          [0,  BG, BG],
                          [BG, BG, BG]])
        lb[1] = np.array([[BG, BG, BG],
                          [BG, 0,   1],
                          [BG, BG, BG]])

        with expected_warnings(['`background`']):
            assert_array_equal(label(x), lnb)

        assert_array_equal(label(x, background=0), lb)

    def test_background_two_regions(self):
        x = np.zeros((2, 3, 3), int)
        x[0] = np.array([[0, 0, 6],
                         [0, 0, 6],
                         [5, 5, 5]])
        x[1] = np.array([[6, 6, 0],
                         [5, 0, 0],
                         [0, 0, 0]])
        lb = x.copy()
        lb[0] = np.array([[BG, BG, 0],
                          [BG, BG, 0],
                          [1,   1, 1]])
        lb[1] = np.array([[0,  0,  BG],
                          [1,  BG, BG],
                          [BG, BG, BG]])

        res = label(x, background=0)
        assert_array_equal(res, lb)

    def test_background_one_region_center(self):
        x = np.zeros((3, 3, 3), int)
        x[1, 1, 1] = 1

        lb = np.ones_like(x) * BG
        lb[1, 1, 1] = 0

        assert_array_equal(label(x, neighbors=4, background=0), lb)

    def test_return_num(self):
        x = np.array([[1, 0, 6],
                      [0, 0, 6],
                      [5, 5, 5]])

        with expected_warnings(['`background`']):
            assert_array_equal(label(x, return_num=True)[1], 4)

        assert_array_equal(label(x, background=0, return_num=True)[1], 3)

    def test_1D(self):
        x = np.array((0, 1, 2, 2, 1, 1, 0, 0))
        xlen = len(x)
        y = np.array((0, 1, 2, 2, 3, 3, 4, 4))
        reshapes = ((xlen,),
                    (1, xlen), (xlen, 1),
                    (1, xlen, 1), (xlen, 1, 1), (1, 1, xlen))
        for reshape in reshapes:
            x2 = x.reshape(reshape)
            with expected_warnings(['`background`']):
                labelled = label(x2)
            assert_array_equal(y, labelled.flatten())

    def test_nd(self):
        x = np.ones((1, 2, 3, 4))
        np.testing.assert_raises(NotImplementedError, label, x)


class TestSupport:
    def test_reshape(self):
        shapes_in = ((3, 1, 2), (1, 4, 5), (3, 1, 1), (2, 1), (1,))
        for shape in shapes_in:
            shape = np.array(shape)
            numones = sum(shape == 1)
            inp = np.random.random(shape)

            fixed, swaps = ccomp.reshape_array(inp)
            shape2 = fixed.shape
            # now check that all ones are at the beginning
            for i in range(numones):
                assert shape2[i] == 1

            back = ccomp.undo_reshape_array(fixed, swaps)
            # check that the undo works as expected
            assert_array_equal(inp, back)


if __name__ == "__main__":
    run_module_suite()
