import numpy as np
from pytest import fixture
from skimage import data
from skimage.measure import regionprops, trace_boundary
from skimage._shared.testing import assert_array_equal, fetch


@fixture(scope="session")
def expected():
    return np.load(fetch("data/boundary_tracing_tests.npz"))


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
    regions = regionprops(example)
    boundary_1 = trace_boundary(regions[0].coords)

    assert_array_equal(boundary_1, expected_boundary)


def test_horse_example(expected):
    horse = np.logical_not(data.horse()).astype(int)
    regions = regionprops(horse)
    boundary = trace_boundary(regions[0].coords)

    # Suggestion:
    # Use erosion for testing
    # https://github.com/scikit-image/scikit-image/pull/4165/files#diff-bfeb06dd4f95632ee7f7e3ae971d0065R370

    assert_array_equal(boundary, expected["horse"])
