from skimage.data import astronaut
from skimage.feature.sift import SIFT
from skimage.color import rgb2gray
from skimage import img_as_float

def test_sift():
    img = img_as_float(rgb2gray(astronaut()))
    s = SIFT(c_edge=13)
    s.detect(img)