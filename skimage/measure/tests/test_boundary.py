import numpy as np
from skimage.measure import regionprops, trace_boundary
from skimage._shared.testing import assert_array_equal

def test_simple_example():
    example = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0],
        ]
    )

    expected_boundary = np.array(
        [[1, 1], [1, 2], [1, 3], [1, 4], [2, 4], [3, 4], [4, 3], [3, 2], [2, 1]]
    )
    props = regionprops(example)
    boundary_1 = trace_boundary(props[0].coords)

    assert_array_equal(boundary_1, expected_boundary)
