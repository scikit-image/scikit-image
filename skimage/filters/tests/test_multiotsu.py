from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage import data

from skimage._shared import testing
from skimage._shared.testing import assert_equal


def test_bimodal_hist():
    """
    """

    image = data.camera()
    thr_otsu = threshold_otsu(image)
    thr_multi, _ = threshold_multiotsu(image, nclass=2)

    assert thr_otsu == thr_multi
