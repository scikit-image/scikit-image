import numpy as np
from skimage import measure
from .test_regionprops import SAMPLE


def test_expanded_convex_hull():
    coords = np.transpose(np.nonzero(SAMPLE))
    vol = measure.extended_convex_hull(coords).volume
    np.testing.assert_allclose(vol, 129.5)
