import numpy as np
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_almost_equal
from skimage._shared.testing import assert_array_equal
from scipy import ndimage as ndi
from skimage.filters import correlate_sparse

def test_correlate_sparse():
    image = np.array([[0, 0, 1, 3, 5],
                      [0, 1, 4, 3, 4],
                      [1, 2, 5, 4, 1],
                      [2, 4, 5, 2, 1],
                      [4, 5, 1, 0, 0]], dtype=np.float)

    kernel = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128]).reshape((3, 3))

    cs_output = correlate_sparse(image, kernel, mode="valid")
    ndi_output = ndi.correlate(image, kernel, mode='wrap')
    ndi_output = ndi_output[1:4, 1:4]

    assert_equal(cs_output, ndi_output)