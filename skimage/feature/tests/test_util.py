import numpy as np
from numpy.testing import assert_array_equal
from skimage.feature.util import pairwise_hamming_distance


def test_pairwise_hamming_distance_range():
    """Values of all the pairwise hamming distances should be in the range
    [0, 1]."""
    a = np.random.random_sample((10, 50)) > 0.5
    b = np.random.random_sample((20, 50)) > 0.5
    dist = pairwise_hamming_distance(a, b)
    assert np.all((0 <= dist) & (dist <= 1))


def test_pairwise_hamming_distance_value():
    """The result of pairwise_hamming_distance of two fixed sets of boolean
    vectors should be same as expected."""
    np.random.seed(10)
    a = np.random.random_sample((4, 100)) > 0.5
    np.random.seed(20)
    b = np.random.random_sample((3, 100)) > 0.5
    result = pairwise_hamming_distance(a, b)
    expected = np.array([[0.5 ,  0.49,  0.44],
                         [0.44,  0.53,  0.52],
                         [0.4 ,  0.55,  0.5 ],
                         [0.47,  0.48,  0.57]])
    assert_array_equal(result, expected)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
