"""Test for correctness of color distance functions

Authors
-------
Matt Terry

:license: modified BSD
"""
from os.path import abspath, dirname, join as pjoin

import numpy as np
from numpy.testing import assert_array_almost_equal

from skimage.color import deltaE_ciede2000


def test_ciede2000_dE():
    dtype = [('pair', int),
             ('1', int),
             ('L1', float),
             ('a1', float),
             ('b1', float),
             ('a1_prime', float),
             ('C1_prime', float),
             ('h1_prime', float),
             ('hbar_prime', float),
             ('G', float),
             ('T', float),
             ('SL', float),
             ('SC', float),
             ('SH', float),
             ('RT', float),
             ('dE', float),
             ('2', int),
             ('L2', float),
             ('a2', float),
             ('b2', float),
             ('a2_prime', float),
             ('C2_prime', float),
             ('h2_prime', float),
             ]

    # note: ciede_test_data.txt contains several intermediate quantities
    path = pjoin(dirname(abspath(__file__)), 'ciede_test_data.txt')
    data = np.loadtxt(path, dtype=dtype)

    N = len(data)

    lab1 = np.zeros((N, 3))
    lab1[:, 0] = data['L1']
    lab1[:, 1] = data['a1']
    lab1[:, 2] = data['b1']

    lab2 = np.zeros((N, 3))
    lab2[:, 0] = data['L2']
    lab2[:, 1] = data['a2']
    lab2[:, 2] = data['b2']

    dE2 = deltaE_ciede2000(lab1, lab2)

    assert_array_almost_equal(dE2, data['dE'])
