import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from skimage.feature import saliency_kadir_brady
from skimage.color import rgb2gray
from skimage.data import astronaut


def test_kadir_brady():
    
    image = astronaut()
    img3 = np.ones((5, 5, 5))

    regions = saliency_kadir_brady(rgb2gray(image))

    scale_range = [(5, 7),(7, 9),(9, 11),(11, 13)]

    # regions categorized under different scale ranges
    s_regions = [[r[0] for r in regions if int(r[2]) in range(i,j)] for i,j in scale_range]

    assert_array_equal(len(s_regions[0]), 0)
    assert_array_equal(len(s_regions[1]), 57)
    assert_array_equal(len(s_regions[2]), 32)
    assert_array_equal(len(s_regions[3]), 0)

    assert_raises(ValueError, saliency_kadir_brady, img3)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
