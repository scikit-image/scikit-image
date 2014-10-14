import numpy as np
from numpy.testing import assert_array_equal
from skimage.saliency import misc
from scipy.ndimage import filters

test_image = np.random.rand(300,400,3) 

def test_lab_saliency():
    out1 = misc.lab_saliency(test_image,5,False)
    test_image_smooth = filters.gaussian_filter(test_image,5)
    out2 = misc.lab_saliency(test_image_smooth, 0, False)
    np.testing.assert_array_equal(out1,out2)

if __name__ == "__main__":
    np.testing.run_module_suite()
