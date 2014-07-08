# -*- coding: utf-8 -*-

import pytest

import numpy as np

np.set_printoptions(edgeitems=10)

from skimage.measure import label_match

@pytest.mark.parametrize("arr1, arr2, dout1, dout2",
                         [
                         # Simple relabel
                         ([1,1,0,0,0,2,2],
                          [0,3,3,0,4,5,5],

                          [1,1,0,0,0,2,2],
                          [0,1,1,0,4,2,2]),
                         # Bit odder
#                         ([1,1,1,1,1,1,1], [0,3,3,0,4,5,5],
#                          [1,1,1,1,1,1,1], [0,1,1,0,4,5,5]),
                         ]
                        )
def test_relabel(arr1, arr2, dout1, dout2):
    out1, out2 = label_match(arr1, arr2, relabel=True, background=0)
    
    np.testing.assert_almost_equal(out1, dout1)
    np.testing.assert_almost_equal(out2, dout2)



@pytest.mark.parametrize("arr1, arr2, dout1, dout2",
                         [
                         # Simple relabel
                         ([1,1,0,0,0,2,2],
                          [0,3,3,0,4,5,5],

                          [1,1,0,0,0,2,2],
                          [0,1,1,0,0,2,2]),
                         # Bit odder
#                         ([1,1,1,1,1,1,1], [0,3,3,0,4,5,5],
#                          [1,1,1,1,1,1,1], [0,1,1,0,4,5,5]),
                         ]
                        )
def test_relabel_nonoverlap(arr1, arr2, dout1, dout2):
    out1, out2 = label_match(arr1, arr2, relabel=True, remove_nonoverlap=True, background=0)
    
    np.testing.assert_almost_equal(out1, dout1)
    np.testing.assert_almost_equal(out2, dout2)

@pytest.mark.parametrize("arr1, arr2, dout1, dout2",
                         [
                         # Simple relabel
                         ([0,0,1,1,1,1,0],
                          [1,1,1,0,2,2,2],

                          [0,0,1,1,1,1,0],
                          [0,0,0,0,1,1,1]),
                         # if multiples with same overlap, keep only the first
                         ([1,1,1,1,1,1,1],
                          [0,3,3,0,4,5,5],

                          [1,1,1,1,1,1,1],
                          [0,1,1,0,0,0,0]),

                         # Edge Cases
                         # Duplicate overlap
                         ([1,1,0,2,2],
                          [1,1,1,1,1],

                          [1,1,0,0,0],
                          [1,1,1,1,1]),
                         # irregular overlap
#                         ([1,1,0,2,2,2,0,3,3,3,3],
#                          [0,1,1,1,0,2,2,2,2,2,2],
#
#                          [1,1,0,0,0,0,0,2,2,2,2],
#                          [0,1,1,1,0,2,2,2,2,2,2]),
                         # irregular overlap
                         ([1,1,1,0,2,2,2,0,3,3,3,0,4],
                          [5,0,4,4,4,0,3,3,3,0,2,2,2],

                          [1,1,1,0,2,2,2,0,0,0,0,0,3],
                          [1,0,0,0,0,0,2,2,2,0,3,3,3]),
                          
                         ]
                        )
def test_relabel_duplicates(arr1, arr2, dout1, dout2):
    out1, out2 = label_match(arr1, arr2, relabel=True, remove_duplicates=True, background=0)
    
    np.testing.assert_almost_equal(out1, dout1)
    np.testing.assert_almost_equal(out2, dout2)


if __name__ == "__main__":
    pytest.main("-x test_relabel.py --tb=short -s")