import numpy as np
from skimage._shared import testing
from skimage.registration import optical_flow_tvl1


def test_no_motion_2d():
    rnd = np.random.RandomState(0)
    img = rnd.normal(size=(256, 256))

    flow = optical_flow_tvl1(img, img)

    assert np.all(flow == 0)


def test_no_motion_3d():
    rnd = np.random.RandomState(0)
    img = rnd.normal(size=(128, 128, 128))

    flow = optical_flow_tvl1(img, img)

    assert np.all(flow == 0)


def test_incompatible_shapes():
    rnd = np.random.RandomState(0)
    I0 = rnd.normal(size=(256, 256))
    I1 = rnd.normal(size=(255, 256))
    with testing.raises(ValueError):
        u, v = optical_flow_tvl1(I0, I1)
