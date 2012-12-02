from numpy.testing import *
import numpy as np

import skimage.io as io


def test_stack_basic():
    x = np.arange(12).reshape(3, 4)
    io.push(x)

    assert_array_equal(io.pop(), x)


@raises(ValueError)
def test_stack_non_array():
    io.push([[1, 2, 3]])

if __name__ == "__main__":
    run_module_suite()
