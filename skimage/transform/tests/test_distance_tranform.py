import sys
sys.path.append('..')
from distance_transform import multidimensional as md
import numpy as np

def one_d():
    case = [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    out = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    if np.array_equal(md(np.asarray(case)),np.asarray(out)):
        print ("success")
    else:
        print("failed")
        print(md(np.asarray(case)))
        print(np.asarray(out))

def two_d():
    case = [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 1, 0, 1], [0, 0, 1, 1, 1]]
    out = [[5.0, 2.0, 1.0, 1.0, 1.0], [4.0, 1.0, 0.0, 0.0, 0.0], [2.0, 1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 2.0]]
    if np.array_equal(md(np.asarray(case)),np.asarray(out)):
        print ("success")
    else:
        print("failed")
        print(md(np.asarray(case)))
        print(np.asarray(out))

def three_d():
    case = [[[1, 0, 1], [1, 1, 1], [0, 0, 1]], [[1, 1, 0], [0, 1, 1], [0, 1, 1]], [[1, 1, 0], [0, 1, 1], [0, 1, 1]]]
    out = [[[1.0, 0.0, 1.0], [1.0, 1.0, 2.0], [0.0, 0.0, 1.0]], [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 2.0]], [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 4.0]]]
    if np.array_equal(md(np.asarray(case)),np.asarray(out)):
        print ("success")
    else:
        print("failed")
        print(md(np.asarray(case)))
        print(np.asarray(out))

one_d()
two_d()
three_d()












