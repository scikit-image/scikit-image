import numpy as np

def test_threshold_cut():
    arr = np.array((100,100,3),dtype='uint8')
    arr[:50,:50] = 0
    arr[:50,50:] = 1
    arr[50:,50:] = 2
    arr[50:,50:] = 3

    
