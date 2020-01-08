import numpy as np
from skimage._shared import testing
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp


def _sin_flow_gen(image0, max_motion=4.5, npics=5):
    """Generate a synthetic ground truth optical flow with a sinusoid as
      first component.

    Parameters:
    ----
    image0: ndarray
        The base image to be warped.
    max_motion: float
        Maximum flow magnitude.
    npics: int
        Number of sinusoid pics.

    Returns
    -------
    flow, image1 : ndarray
        The synthetic ground truth optical flow with a sinusoid as
        first component and the corresponding warped image.

    """
    grid = np.meshgrid(*[np.arange(n) for n in image0.shape], indexing='ij')
    gt_flow = np.zeros_like(grid)
    gt_flow[0, ...] = max_motion * np.sin(grid[0]/grid[0].max()*npics*np.pi)
    image1 = warp(image0, grid-gt_flow, mode='nearest')
    return gt_flow, image1


def test_2d_motion():
    # Generate synthetic data
    rnd = np.random.RandomState(0)
    image0 = rnd.normal(size=(256, 256))
    gt_flow, image1 = _sin_flow_gen(image0)
    # Estimate the flow
    flow = optical_flow_tvl1(image0, image1, attachment=5)
    # Assert that the average absolute error is less then half a pixel
    assert abs(flow - gt_flow) .mean() < 0.5


def test_3d_motion():
    # Generate synthetic data
    rnd = np.random.RandomState(0)
    image0 = rnd.normal(size=(128, 128, 128))
    gt_flow, image1 = _sin_flow_gen(image0)
    # Estimate the flow
    flow = optical_flow_tvl1(image0, image1, attachment=5)
    # Assert that the average absolute error is less then half a pixel
    assert abs(flow - gt_flow) .mean() < 0.5


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


def test_optical_flow_dtype():
    # Generate synthetic data
    rnd = np.random.RandomState(0)
    image0 = rnd.normal(size=(256, 256))
    gt_flow, image1 = _sin_flow_gen(image0)
    # Estimate the flow at double precision
    flow_f64 = optical_flow_tvl1(image0, image1, attachment=5, dtype=np.float64)

    assert flow_f64.dtype == np.float64

    # Estimate the flow at single precision
    flow_f32 = optical_flow_tvl1(image0, image1, attachment=5, dtype=np.float32)

    assert flow_f32.dtype == np.float32

    # Assert that floating point precision does not affect the quality
    # of the estimated flow

    assert np.abs(flow_f64 - flow_f32).mean() < 1e-3


def test_incompatible_shapes():
    rnd = np.random.RandomState(0)
    I0 = rnd.normal(size=(256, 256))
    I1 = rnd.normal(size=(128, 256))
    with testing.raises(ValueError):
        u, v = optical_flow_tvl1(I0, I1)


def test_wrong_dtype():
    rnd = np.random.RandomState(0)
    img = rnd.normal(size=(256, 256))
    with testing.raises(ValueError):
        u, v = optical_flow_tvl1(img, img, dtype=np.int64)
