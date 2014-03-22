import numpy as np
from numpy.testing import assert_array_equal
from skimage.saliency import spectral_residue
from skimage.color import rgb2gray

test_image = np.random.rand(300, 400, 3) 

def test_sr_saliency():
    out1 = spectral_residue.sr_saliency(test_image, 3, False)
    out2 = spectral_residue.sr_saliency(rgb2gray(test_image), 3, False)
    np.testing.assert_array_equal(out1, out2)

if __name__ == "__main__":
    np.testing.run_module_suite()
