import numpy as np
from skimage.registration import tvl1


def test_no_motion():
    rnd = np.random.RandomState(0)
    img = rnd.normal(size=(256, 256))

    u, v = tvl1(img, img)

    assert np.all(u == 0)
    assert np.all(u == 0)
