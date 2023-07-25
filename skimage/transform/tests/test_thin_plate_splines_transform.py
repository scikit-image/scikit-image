import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from skimage.transform._thin_plate_splines import (TpsTransform, _ensure_2d,
                                                   tps_warp)

SRC = np.array([
    [0, 0],
    [0, 5],
    [5, 5],
    [5, 0]
])

DST = np.array([
    [5, 0],
    [0, 0],
    [0, 5],
    [5, 5]
])


def test_tps_transform_inverse():
    tps = TpsTransform()
    with pytest.raises(NotImplementedError):
        tps.inverse()


def test_tps_transform_ensure_2d():
    assert_array_equal(_ensure_2d(SRC), SRC)
    assert_array_equal(_ensure_2d(DST), DST)

    array_1d = np.array([0, 5, 10])
    expected = np.array([[0], [5], [10]])
    assert_array_equal(_ensure_2d(array_1d), expected)

    empty_array = np.array([])
    with pytest.raises(ValueError, match="Array of points can not be empty."):
        _ensure_2d(empty_array)

    scalar = 5
    with pytest.raises(ValueError, match="Array must be be 2D."):
        _ensure_2d(scalar)

    array_3d = np.array([[[0, 5], [10, 15]], [[20, 25], [30, 35]]])
    with pytest.raises(ValueError, match="Array must be be 2D."):
        _ensure_2d(array_3d)

    control_pts_less_than_3 = np.array([[0, 0], [0, 0]])
    with pytest.raises(ValueError, match="Array points less than 3 is undefined."):
        _ensure_2d(control_pts_less_than_3)

def test_tps_transform_init():
    tform = TpsTransform()

    # Test that _estimated is initialized to False
    assert tform._estimated is False
    assert tform.parameters is None
    assert tform.control_points is None

def test_tps_transform_estimation():
    tform = TpsTransform()

    # Ensure that the initial state is as expected
    assert tform._estimated is False
    assert tform.parameters is None
    assert tform.control_points is None

    # Perform estimation
    tcoeffs = tform.estimate(SRC, DST)

    # Check if the estimation was successful
    assert tcoeffs
    assert tform._estimated is True
    assert len(tform.control_points) > 0

    assert len(tform.parameters) > 0
    assert tform.parameters.shape[0] == SRC.shape[0] + 3
    np.testing.assert_array_equal(tform.control_points, SRC)

def test_tps_transform_estimation_failure():
    # Test the estimate method when the estimation fails
    tform = TpsTransform()
    src = np.array([[0, 0], [0, 5], [5, 5], [5, 0]])
    dst = np.array([[5, 0], [0, 0], [0, 5]])

    # Ensure that the initial state is as expected
    assert tform._estimated is False
    assert tform.parameters is None
    assert tform.control_points is None

    # Perform the estimation, which should fail due to the mismatched number of points
    with pytest.raises(ValueError):
        tform.estimate(src, dst)

    # Check if the estimation failed and the instance attributes remain unchanged
    assert tform._estimated is False
    assert tform.parameters is None
    assert tform.control_points is None

@pytest.mark.parametrize('image_shape', [0, (0, 10), (10, 0)])
def test_zero_image_size(image_shape):
    tform = TpsTransform()
    tform.estimate(SRC, DST)
    img = np.zeros(image_shape)

    with pytest.raises(ValueError):
        tps_warp(img, SRC, DST)
    with pytest.raises(ValueError):
        tps_warp(img, SRC, DST)
    with pytest.raises(ValueError):
        tps_warp(img, SRC, DST)

def test_tps_transform_call():
    # Test __call__ method without esitmate
    tform = TpsTransform()
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 2, 3, 4])

    with pytest.raises(ValueError, match="None. Compute the `estimate`"):
        tform(x, y)

    # Test __call__ method with estimmate
    tform.estimate(SRC, DST)
    xx, yy = np.meshgrid(np.arange(5), np.arange(5))
    xx_trans, yy_trans = tform(xx, yy)

    expected_xx = np.array([[5., 5., 5., 5., 5.],
                            [4., 4., 4., 4., 4.],
                            [3., 3., 3., 3., 3.],
                            [2., 2., 2., 2., 2.],
                            [1., 1., 1., 1., 1.]])
    assert_allclose(xx_trans, expected_xx)


def test_tps_warp_resizing():
    pass

def test_tps_warp_rotation():
    pass

def test_tps_warp_translation():
    pass
