import numpy as np
from nose.tools import raises
from numpy.testing import assert_equal
from skimage.util import FuncExec, AsyncPoolExec

def test_func_exec_result():
    A = np.arange(3*2*1*2).reshape(3,2,1,2)
    fe = FuncExec(np.sum, {'axis': 1})
    assert_equal(fe(A).result(), np.sum(A, axis=3))

def test_func_exec_ready():
    A = np.arange(3*2*1*2).reshape(3,2,1,2)
    fe = FuncExec(np.sum, {'axis': 0})
    results = {
        (0, 0): np.array([0, 1]),
        (0, 1): np.array([2, 3]),
        (1, 0): np.array([4, 5]),
        (1, 1): np.array([6, 7]),
        (2, 0): np.array([8, 9]),
        (2, 1): np.array([10, 11])
    }
    for result in fe(A).ready():
        print result[0], result[1]
        assert_equal(result[1], results[result[0]])

def test_async_pool_exec_result():
    A = np.arange(3*2*1*2).reshape(3,2,1,2)
    fe = AsyncPoolExec(np.sum, {'axis': 1})
    assert_equal(fe(A).result(), np.sum(A, axis=3))

def test_async_pool_exec_ready():
    A = np.arange(3*2*1*2).reshape(3,2,1,2)
    fe = AsyncPoolExec(np.sum, {'axis': 0}, pool_size=4)
    results = {
        (0, 0): np.array([0, 1]),
        (0, 1): np.array([2, 3]),
        (1, 0): np.array([4, 5]),
        (1, 1): np.array([6, 7]),
        (2, 0): np.array([8, 9]),
        (2, 1): np.array([10, 11])
    }
    for result in fe(A).ready():
        print result[0], result[1]
        assert_equal(result[1], results[result[0]])

    
if __name__ == '__main__':
    np.testing.run_module_suite()
