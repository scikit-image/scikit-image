import numpy as np

from skimage import data
from skimage.util.crop import crop

def test_2d_crop_1():
    data = np.random.random((50, 50))
    out_data = crop(data, [(0, 25)])
    assert out_data.shape == (25, 50)

def test_2d_crop_2():
    out_data = crop(data, [(0, 25)], axis=[1])
    assert out_data.shape == (50, 25)
    
def test_2d_crop_3():
    out_data = crop(data, [(0, 25), (0, 30)], axis=[1, 0])
    assert out_data.shape == (30, 25)

def test_nd_crop():
    data = np.random.random((50, 50, 50))
    out_data = crop(data, [(0, 25)])
    assert out_data.shape == (25, 50, 50)



