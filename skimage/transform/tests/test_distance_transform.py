from skimage.transform.distance_transform import (generalized_distance_transform,
                                                  manhattan_dist,
                                                  manhattan_meet)
import numpy as np
from warnings import warn


def test_1d():
    case = [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    out_euc = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    out_man = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case),func="slow"),np.asarray(out_euc))
    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case),func="slow", dist_func=manhattan_dist, dist_meet=manhattan_meet),np.asarray(out_man))


def test_2d():
    case = [[1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 1, 0, 1],
            [0, 0, 1, 1, 1]]

    out_euc = [[5.0, 2.0, 1.0, 1.0, 1.0],
               [4.0, 1.0, 0.0, 0.0, 0.0],
               [2.0, 1.0, 1.0, 0.0, 0.0],
               [1.0, 0.0, 1.0, 0.0, 1.0],
               [0.0, 0.0, 1.0, 1.0, 2.0]]

    out_man = [[3.0, 2.0, 1.0, 1.0, 1.0],
               [2.0, 1.0, 0.0, 0.0, 0.0],
               [2.0, 1.0, 1.0, 0.0, 0.0],
               [1.0, 0.0, 1.0, 0.0, 1.0],
               [0.0, 0.0, 1.0, 1.0, 2.0]]

    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case),func="slow"),np.asarray(out_euc))
    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case),func="slow", dist_func=manhattan_dist, dist_meet=manhattan_meet),np.asarray(out_man))


def test_3d():
    case = [[[1, 0, 1],
             [1, 1, 1],
             [0, 0, 1]],
            [[1, 1, 0],
             [0, 1, 1],
             [0, 1, 1]],
            [[1, 1, 0],
             [0, 1, 1],
             [0, 1, 1]]]

    out_euc = [[[1.0, 0.0, 1.0],
                [1.0, 1.0, 2.0],
                [0.0, 0.0, 1.0]],
               [[1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 2.0]],
               [[1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 4.0]]]

    out_man = [[[1.0, 0.0, 1.0],
                [1.0, 1.0, 2.0],
                [0.0, 0.0, 1.0]],
               [[1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 2.0]],
               [[1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 2.0]]]

    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case),func="slow"),np.asarray(out_euc))
    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case),func="slow", dist_func=manhattan_dist, dist_meet=manhattan_meet),np.asarray(out_man))

def test_euclidean_equivalent_to_ndimage():
    from scipy.ndimage.morphology import distance_transform_edt
    case = (1+-1*(np.random.randint(50, size=(256,256,256))//49)).astype('float64')

    out_euc = distance_transform_edt(case)
    skimage_out_euc = generalized_distance_transform(case, func='euclidean')**0.5

    np.testing.assert_allclose(skimage_out_euc,out_euc)

def test_manhattan_equivalent_to_ndimage():
    from scipy.ndimage.morphology import distance_transform_cdt
    case = (1+-1*(np.random.randint(50, size=(256,256,256))//49)).astype('float64')

    out_man = distance_transform_cdt(case, metric = 'taxicab')
    skimage_out_man = generalized_distance_transform(case, func='manhattan')

    np.testing.assert_allclose(skimage_out_man,out_man)
