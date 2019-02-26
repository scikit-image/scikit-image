from skimage.transform.distance_transform import generalized_distance_transform
import numpy as np

def test_1d():
    case = [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    out = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case)),np.asarray(out))

def test_2d():
    case = [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 1, 0, 1], [0, 0, 1, 1, 1]]
    out = [[5.0, 2.0, 1.0, 1.0, 1.0], [4.0, 1.0, 0.0, 0.0, 0.0], [2.0, 1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 2.0]]
    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case)),np.asarray(out))

def test_3d():
    case = [[[1, 0, 1], [1, 1, 1], [0, 0, 1]], [[1, 1, 0], [0, 1, 1], [0, 1, 1]], [[1, 1, 0], [0, 1, 1], [0, 1, 1]]]
    out = [[[1.0, 0.0, 1.0], [1.0, 1.0, 2.0], [0.0, 0.0, 1.0]], [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 2.0]], [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 4.0]]]
    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case)),np.asarray(out))
