from skimage.transform.distance_transform import (generalized_distance_transform,
                                                  manhattan_dist,
                                                  manhattan_meet)
import numpy as np
from warnings import warn


def test_1d():
    case = [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    out_euc = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    out_man = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case)),np.asarray(out_euc))
    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case), dist_func=manhattan_dist, dist_meet=manhattan_meet),np.asarray(out_man))


def test_2d():
    case = [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 1, 0, 1], [0, 0, 1, 1, 1]]
    out_euc = [[5.0, 2.0, 1.0, 1.0, 1.0], [4.0, 1.0, 0.0, 0.0, 0.0], [2.0, 1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 2.0]]
    out_man = [[3.0, 2.0, 1.0, 1.0, 1.0], [2.0, 1.0, 0.0, 0.0, 0.0], [2.0, 1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 2.0]]
    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case)),np.asarray(out_euc))
    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case), dist_func=manhattan_dist, dist_meet=manhattan_meet),np.asarray(out_man))


def test_3d():
    case = [[[1, 0, 1], [1, 1, 1], [0, 0, 1]], [[1, 1, 0], [0, 1, 1], [0, 1, 1]], [[1, 1, 0], [0, 1, 1], [0, 1, 1]]]
    out_euc = [[[1.0, 0.0, 1.0], [1.0, 1.0, 2.0], [0.0, 0.0, 1.0]], [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 2.0]], [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 4.0]]]
    out_man = [[[1.0, 0.0, 1.0], [1.0, 1.0, 2.0], [0.0, 0.0, 1.0]], [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 2.0]], [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 2.0]]]
    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case)),np.asarray(out_euc))
    np.testing.assert_allclose(generalized_distance_transform(np.asarray(case), dist_func=manhattan_dist, dist_meet=manhattan_meet),np.asarray(out_man))

def test_large():
    from scipy.ndimage.morphology import distance_transform_edt, distance_transform_cdt
    import time
    case = np.random.randint(2, size=(10)) #.astype('float64')

    start = time.time()
    out_euc = distance_transform_edt(case)**2
    out_man = distance_transform_cdt(case, metric = 'taxicab')
    print('scipy time:', time.time()-start)

    start = time.time()
    skimage_out_euc = generalized_distance_transform(case, func='slow')
    skimage_out_man = generalized_distance_transform(case, func='manhattan')
    print('skimage time:', time.time()-start)
    warn(str(skimage_out_man.tolist()))
    #np.testing.assert_allclose(skimage_out_euc,out_euc)
    np.testing.assert_allclose(skimage_out_man,out_man)

