import numpy as np
from skimage.draw import circle
from skimage.feature import get_blobs
import math

def test_get_blobs():
    img = np.ones((256,256))
    xs,ys = circle(20,30,15)
    img[xs,ys] = 255
    x,y,a = get_blobs(img,min_sigma = 1,max_sigma=10)[0]
    
    assert abs(x-20) <= 2
    assert abs(y-30) <= 2
    assert abs(math.sqrt(a/math.pi) - 15) <= 2
