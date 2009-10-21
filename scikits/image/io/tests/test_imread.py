import os.path
import numpy as np

from scikits.image import data_dir
from scikits.image.io import imread

def test_imread_flatten():
    # a color image is flattened and returned as float32
    img = imread(os.path.join(data_dir, 'color.png'), flatten=True)
    assert img.dtype == np.float32
    img = imread(os.path.join(data_dir, 'camera.png'), flatten=True)
    # check that flattening does not occur for an image that is grey already.
    assert np.sctype2char(img.dtype) in np.typecodes['AllInteger']

def test_imread_dtype():
    img = imread(os.path.join(data_dir, 'camera.png'), dtype=np.float64)
    assert img.dtype == np.float64
