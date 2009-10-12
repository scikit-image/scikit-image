import os.path
import numpy as np

from scikits.image import data_dir
from scikits.image.io import imread

def test_imread():
    img = imread(os.path.join(data_dir, 'camera.png'), dtype=np.float32)
    print img.dtype, type(img)
    assert img.dtype == np.float32
