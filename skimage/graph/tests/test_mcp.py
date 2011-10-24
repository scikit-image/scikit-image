import numpy as np
from numpy.testing import *

import skimage.graph.mcp as mcp

a = np.ones((8,8), dtype=np.float32)
a[1:-1, 1] = 0
a[1, 1:-1] = 0

## array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
##        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
##        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
##        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
##        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
##        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
##        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
##        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]], dtype=float32)

def test_basic():
    m = mcp.MCP(a, fully_connected=True)
    costs, traceback = m.find_costs([(1,6)])
    return_path = m.traceback((7, 2))
    assert_array_equal(costs,
                       [[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
                        [ 1.,  0.,  1.,  2.,  2.,  2.,  2.,  2.],
                        [ 1.,  0.,  1.,  2.,  3.,  3.,  3.,  3.],
                        [ 1.,  0.,  1.,  2.,  3.,  4.,  4.,  4.],
                        [ 1.,  0.,  1.,  2.,  3.,  4.,  5.,  5.],
                        [ 1.,  1.,  1.,  2.,  3.,  4.,  5.,  6.]])

    assert_array_equal(return_path,
                       [(1, 6),
                        (1, 5),
                        (1, 4),
                        (1, 3),
                        (1, 2),
                        (2, 1),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                        (7, 2)])

def test_neg_inf():
    expected_costs = np.where(a==1, np.inf, 0)
    expected_path = [(1, 6),
                     (1, 5),
                     (1, 4),
                     (1, 3),
                     (1, 2),
                     (2, 1),
                     (3, 1),
                     (4, 1),
                     (5, 1),
                     (6, 1)]
    test_neg = np.where(a==1, -1, 0)
    test_inf = np.where(a==1, np.inf, 0)
    m = mcp.MCP(test_neg, fully_connected=True)
    costs, traceback = m.find_costs([(1, 6)])
    return_path = m.traceback((6, 1))
    assert_array_equal(costs, expected_costs)
    assert_array_equal(return_path, expected_path)
    m = mcp.MCP(test_inf, fully_connected=True)
    costs, traceback = m.find_costs([(1, 6)])
    return_path = m.traceback((6, 1))
    assert_array_equal(costs, expected_costs)
    assert_array_equal(return_path, expected_path)
  

def test_route():
    return_path, cost = mcp.route_through_array(a, (1,6), (7,2), geometric=True)
    assert_almost_equal(cost, np.sqrt(2)/2)
    assert_array_equal(return_path,
                       [(1, 6),
                        (1, 5),
                        (1, 4),
                        (1, 3),
                        (1, 2),
                        (2, 1),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                        (7, 2)])

def test_no_diagonal():
    m = mcp.MCP(a, fully_connected=False)
    costs, traceback = m.find_costs([(1,6)])
    return_path = m.traceback((7, 2))
    assert_array_equal(costs,
                       [[ 2.,  1.,  1.,  1.,  1.,  1.,  1.,  2.],
                        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  2.],
                        [ 1.,  0.,  1.,  2.,  2.,  2.,  2.,  3.],
                        [ 1.,  0.,  1.,  2.,  3.,  3.,  3.,  4.],
                        [ 1.,  0.,  1.,  2.,  3.,  4.,  4.,  5.],
                        [ 1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.],
                        [ 2.,  1.,  2.,  3.,  4.,  5.,  6.,  7.]])
    assert_array_equal(return_path,
                       [(1, 6),
                        (1, 5),
                        (1, 4),
                        (1, 3),
                        (1, 2),
                        (1, 1),
                        (2, 1),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                        (7, 1),
                        (7, 2)])


def test_crashing():
    for shape in [(100, 100), (5, 8, 13, 17)]:
        yield _test_random, shape

def _test_random(shape):
    # Just tests for crashing -- not for correctness.
    np.random.seed(0)
    a = np.random.random(shape).astype(np.float32)
    starts = [[0]*len(shape), [-1]*len(shape),
              (np.random.random(len(shape))*shape).astype(int)]
    ends = [(np.random.random(len(shape))*shape).astype(int) for i in range(4)]
    m = mcp.MCP(a, fully_connected=True)
    costs, offsets = m.find_costs(starts)
    for point in [(np.random.random(len(shape))*shape).astype(int)
                  for i in range(4)]:
        m.traceback(point)
    m._reset()
    m.find_costs(starts, ends)
    for end in ends:
        m.traceback(end)
    return a, costs, offsets


if __name__ == "__main__":
    run_module_suite()
