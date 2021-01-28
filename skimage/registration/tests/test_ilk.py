import numpy as np
import pytest
from skimage._shared import testing
from skimage.registration import optical_flow_ilk
from skimage.transform import warp
from test_tvl1 import _sin_flow_gen


@pytest.mark.parametrize('gaussian', [True, False])
@pytest.mark.parametrize('prefilter', [True, False])
def test_2d_motion(gaussian, prefilter):
    # Generate synthetic data
    rnd = np.random.RandomState(0)
    image0 = rnd.normal(size=(256, 256))
    gt_flow, image1 = _sin_flow_gen(image0)
    # Estimate the flow
    flow = optical_flow_ilk(image0, image1,
                            gaussian=gaussian, prefilter=prefilter)
    # Assert that the average absolute error is less then half a pixel
    assert abs(flow - gt_flow).mean() < 0.5


@pytest.mark.parametrize('gaussian', [True, False])
@pytest.mark.parametrize('prefilter', [True, False])
def test_3d_motion(gaussian, prefilter):
    # Generate synthetic data
    rnd = np.random.RandomState(0)
    image0 = rnd.normal(size=(50, 55, 60))
    gt_flow, image1 = _sin_flow_gen(image0, npics=3)
    # Estimate the flow
    flow = optical_flow_ilk(image0, image1, radius=5,
                            gaussian=gaussian, prefilter=prefilter)

    # Assert that the average absolute error is less then half a pixel
    assert abs(flow - gt_flow).mean() < 0.5


def test_no_motion_2d():
    rnd = np.random.RandomState(0)
    img = rnd.normal(size=(256, 256))

    flow = optical_flow_ilk(img, img)

    assert np.all(flow == 0)


def test_no_motion_3d():
    rnd = np.random.RandomState(0)
    img = rnd.normal(size=(64, 64, 64))

    flow = optical_flow_ilk(img, img)

    assert np.all(flow == 0)


def test_optical_flow_dtype():
    # Generate synthetic data
    rnd = np.random.RandomState(0)
    image0 = rnd.normal(size=(256, 256))
    gt_flow, image1 = _sin_flow_gen(image0)
    # Estimate the flow at double precision
    flow_f64 = optical_flow_ilk(image0, image1, dtype='float64')

    assert flow_f64.dtype == 'float64'

    # Estimate the flow at single precision
    flow_f32 = optical_flow_ilk(image0, image1, dtype='float32')

    assert flow_f32.dtype == 'float32'

    # Assert that floating point precision does not affect the quality
    # of the estimated flow

    assert abs(flow_f64 - flow_f32).mean() < 1e-3


def test_incompatible_shapes():
    rnd = np.random.RandomState(0)
    I0 = rnd.normal(size=(256, 256))
    I1 = rnd.normal(size=(255, 256))
    with testing.raises(ValueError):
        u, v = optical_flow_ilk(I0, I1)


def test_wrong_dtype():
    rnd = np.random.RandomState(0)
    img = rnd.normal(size=(256, 256))
    with testing.raises(ValueError):
        u, v = optical_flow_ilk(img, img, dtype='int')
