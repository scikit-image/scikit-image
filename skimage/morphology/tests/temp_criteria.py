import math
import unittest

import numpy as np

from skimage.morphology import criteria
from skimage.morphology import extrema
from skimage.measure import label
from scipy import ndimage as ndi
import pdb

import skimage.io
import time

eps = 1e-12


def diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    t = ((a - b)**2).sum()
    return math.sqrt(t)

def test_cameraman():
    filename = '/Users/twalter/data/test/test_images/cameraman.tiff'
    img = skimage.io.imread(filename)
    
    start_time = time.time()
    for s in [5, 10, 15, 20, 25]: 
        open_img = criteria.area_closing(img, s)
    stop_time = time.time()
    
    diff_time = stop_time - start_time
    print 'time elapsed : %.3f' % diff_time
    print 'average time : %.3f' % (diff_time / 5.0)
    return


def test_area_closing2():
    img = np.array(
        [[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
         [240, 200, 200, 240, 200, 240, 200, 200, 240, 240, 200, 240],
         [240, 200,  40, 240, 240, 240, 240, 240, 240, 240,  40, 240],
         [240, 240, 240, 240, 100, 240, 100, 100, 240, 240, 200, 240],
         [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
         [200, 200, 200, 200, 200, 200, 200, 240, 200, 200, 255, 255],
         [200, 255, 200, 200, 200, 255, 200, 240, 255, 255, 255,  40],
         [200, 200, 200, 100, 200, 200, 200, 240, 255, 255, 255, 255],
         [200, 200, 200, 100, 200, 200, 200, 240, 200, 200, 255, 255],
         [200, 200, 200, 200, 200,  40, 200, 240, 240, 100, 255, 255],
         [200,  40, 255, 255, 255,  40, 200, 255, 200, 200, 255, 255],
         [200, 200, 200, 200, 200, 200, 200, 255, 255, 255, 255, 255]], 
        dtype=np.uint8)
    
    seeds_bin = extrema.local_minima(img)
    seeds = label(seeds_bin).astype(np.uint64)
    skimage.io.imsave('/Users/twalter/data/test/test_images/criteria_label.png', seeds)
    
    #output_4 = criteria.area_closing(img, 4)
    output_2 = criteria.area_closing(img, 2)
    expected_2 = np.array(
        [[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
         [240, 200, 200, 240, 240, 240, 200, 200, 240, 240, 200, 240],
         [240, 200, 200, 240, 240, 240, 240, 240, 240, 240, 200, 240],
         [240, 240, 240, 240, 240, 240, 100, 100, 240, 240, 200, 240],
         [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
         [200, 200, 200, 200, 200, 200, 200, 240, 200, 200, 255, 255],
         [200, 255, 200, 200, 200, 255, 200, 240, 255, 255, 255, 255],
         [200, 200, 200, 100, 200, 200, 200, 240, 255, 255, 255, 255],
         [200, 200, 200, 100, 200, 200, 200, 240, 200, 200, 255, 255],
         [200, 200, 200, 200, 200,  40, 200, 240, 240, 200, 255, 255],
         [200, 200, 255, 255, 255,  40, 200, 255, 200, 200, 255, 255],
         [200, 200, 200, 200, 200, 200, 200, 255, 255, 255, 255, 255]],
        dtype=np.uint8)
    expected_4 = np.array(
        [[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
         [240, 200, 200, 240, 240, 240, 240, 240, 240, 240, 240, 240],
         [240, 200, 200, 240, 240, 240, 240, 240, 240, 240, 240, 240],
         [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
         [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
         [200, 200, 200, 200, 200, 200, 200, 240, 240, 240, 255, 255],
         [200, 255, 200, 200, 200, 255, 200, 240, 255, 255, 255, 255],
         [200, 200, 200, 200, 200, 200, 200, 240, 255, 255, 255, 255],
         [200, 200, 200, 200, 200, 200, 200, 240, 200, 200, 255, 255],
         [200, 200, 200, 200, 200, 200, 200, 240, 240, 200, 255, 255],
         [200, 200, 255, 255, 255, 200, 200, 255, 200, 200, 255, 255],
         [200, 200, 200, 200, 200, 200, 200, 255, 255, 255, 255, 255]], 
        dtype=np.uint8)
    
    pdb.set_trace()
    error = diff(output_2, expected_2)
    assert error < eps

    error = diff(output_4, expected_4)
    assert error < eps

    for dt in [np.uint32, np.uint64]: 
        img_cast = img.astype(dt)
        out = criteria.area_closing(img_cast, 2)
        exp_cast = expected_2.astype(dt)
        error = diff(out, exp_cast)
        assert error < eps
        
    data_float = data.astype(np.float64)
    data_float = data_float / 255.0
    for dt in [np.float16, np.float32, np.float64]: 
        data_float = data_float.astype(dt)
        out = criteria.area_closing(data_float, 4)
        exp_cast = expected_4.astype(dt)
        error = diff(out, exp_cast)
        assert error < eps

    img_signed = img.astype(np.int16)
    img_signed = img_signed - 128
    exp_signed = exp_signed.astype(np.int16)
    exp_signed = exp_signed - 128
    for dt in [np.int8, np.int16, np.int32, np.int64]:
        
        img_s = img_signed.astype(dt)
        out = criteria.area_closing 
        exp_s = exp_signed.astype(dt)
        
        error = diff(out, exp_s)
        assert error < eps
        

def test_area_closing():
    "test for area closing"
    data = np.array(
        [[250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
         [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
         [250, 250, 100, 100, 250, 250, 250,  50, 250, 250],
         [250, 250, 100, 100, 250, 250, 250,  50, 250, 250],
         [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
         [250, 250, 250, 250, 250, 255, 255, 255, 255, 255],
         [250, 250, 250, 250, 250, 255, 255, 255, 255, 255],
         [250, 120, 250, 250, 250, 255, 180, 180, 180, 255],
         [250, 250, 250, 250, 250, 255, 180, 180, 180, 255],
         [250, 250, 250, 250, 250, 255, 255, 255, 255, 255]], 
        dtype=np.uint8)
    data_float = data.astype(np.double) / 255.0
    
    output = criteria.area_closing(data_float, 4)

    output_8bit = 255.0 * output
    output_8bit = output_8bit.astype(np.uint8)

    expected = np.array(
        [[250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
         [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
         [250, 250, 100, 100, 250, 250, 250, 250, 250, 250],
         [250, 250, 100, 100, 250, 250, 250, 250, 250, 250],
         [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
         [250, 250, 250, 250, 250, 255, 255, 255, 255, 255],
         [250, 250, 250, 250, 250, 255, 255, 255, 255, 255],
         [250, 250, 250, 250, 250, 255, 180, 180, 180, 255],
         [250, 250, 250, 250, 250, 255, 180, 180, 180, 255],
         [250, 250, 250, 250, 250, 255, 255, 255, 255, 255]], 
        dtype=np.uint8)

    error = diff(output_8bit, expected)
    assert error < eps



#test_area_closing()
#test_cameraman()
