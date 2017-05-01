import ext
import numpy as np
import numpy.random as random
import cv
import time
from numpy.testing import *
kernel = np.array([
    [20, 50, 80, 50, 20, 1], 
    [50, 100, 140, 100, 50, 1], 
    [90, 160, 200, 160, 90, 1], 
    [50, 100, 140, 100, 50, 1], 
    [20, 50, 80, 50, 20, 1]], dtype=np.float32)
    

size = (4, 4)
#a = np.ones(size, dtype=np.float32)
#a = np.ones(size, dtype=np.float32)
#a = np.arange(2, size[0]*size[1]+2).reshape(*size).astype(np.float32)
a = random.randn(1000, 1000).astype(np.float32)

b = np.zeros_like(a)
e = np.zeros_like(a)
anchor = (0, 0)


t = time.time()
ext.pyconvolve(a, b, kernel, anchor=anchor)
print "sc", (time.time() - t)*1E3

t = time.time()
cv.Filter2D(a, e, kernel, anchor=anchor)
print "cv", (time.time() - t)*1E3

from scipy.ndimage import convolve
t = time.time()
c = convolve(a, kernel)
print "nd", (time.time() - t)*1E3
k = 1
#print b[0, :]#[5:10,5:10]
#print e[0, :]#[5:10,5:10]
#print b[-1, :]#[5:10,5:10]
#print e[-1,:]#[5:10,5:10]
#print c[5:10,5:10]
print np.sum(b - e)
for i in range(100000):
    kx, ky = [random.randint(1, 10) for k in range(2)]
    #kx, ky = [7, 9]
    image_x, image_y = [random.randint(1, 1000) for k in range(2)]
    #image_x, image_y = [300, 1]
    kernel = random.randn(ky, kx).astype(np.float32)
    image = random.randn(image_y, image_x).astype(np.float32)
    sci_out = np.empty_like(image)
    cv_out = np.empty_like(image)
    anchor = (random.randint(0, kx),random.randint(0, ky))
    print (ky, kx), (image_y, image_x), anchor
    print "sci"
    ext.pyconvolve(image, sci_out, kernel, anchor=anchor)
    print "cv"
    cv.Filter2D(image, cv_out, kernel, anchor=anchor)
    assert_equal(np.sum(sci_out - cv_out), 0)





