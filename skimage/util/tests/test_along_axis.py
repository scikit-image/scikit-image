from skimage.util.along_axis import apply_along_axis
import numpy as np


def test_apply_along_axis():
    test = tuple([np.arange(5) for i in range(5)])
    result = np.array([ 0,  5, 10, 15, 20])
    output = apply_along_axis(sum, 0, test)
    np.testing.assert_allclose(output, result)
