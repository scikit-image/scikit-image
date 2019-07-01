import numpy as np
from skimage._shared import testing
from skimage.registration import tvl1


def test_no_motion():
    rnd = np.random.RandomState(0)
    img = rnd.normal(size=(256, 256))

    u, v = tvl1(img, img)

    assert np.all(u == 0)
    assert np.all(v == 0)


def test_wrong_ndim():
    rnd = np.random.RandomState(0)
    I0 = rnd.normal(size=(256, 256, 3))
    I1 = rnd.normal(size=(256, 256, 3))
    with testing.raises(ValueError):
        u, v = tvl1(I0, I1)


def test_incompatible_shapes():
    rnd = np.random.RandomState(0)
    I0 = rnd.normal(size=(256, 256))
    I1 = rnd.normal(size=(255, 256))
    with testing.raises(ValueError):
        u, v = tvl1(I0, I1)
