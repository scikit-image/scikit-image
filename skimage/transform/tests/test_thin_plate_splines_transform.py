import numpy as np
import pytest
from numpy.testing import assert_array_equal

import skimage as ski
from skimage.transform._thin_plate_splines import (TPSTransform, _ensure_2d,
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


def test_zero_image_size():
    with pytest.raises(ValueError):
        tps_warp(np.zeros(0), TPSTransform())
    with pytest.raises(ValueError):
        tps_warp(np.zeros((0, 10)), TPSTransform())
    with pytest.raises(ValueError):
        tps_warp(np.zeros((10, 0)), TPSTransform())


def test_tps_transform_inverse():
    tps = TPSTransform()
    with pytest.raises(NotImplementedError):
        tps.inverse()


def test_tps_transform_ensure_2d():
    assert_array_equal(_ensure_2d(SRC), SRC)
    assert_array_equal(_ensure_2d(DST), DST)

    array_1d = np.array([0, 5, 10])
    expected = np.array([[0], [5], [10]])
    assert_array_equal(_ensure_2d(array_1d), expected)

    empty_array = np.array([])
    with pytest.raises(ValueError):
        _ensure_2d(empty_array)

    scalar = 5
    with pytest.raises(ValueError):
        _ensure_2d(scalar)

    array_3d = np.array([[[0, 5], [10, 15]], [[20, 25], [30, 35]]])
    with pytest.raises(ValueError):
        _ensure_2d(array_3d)

    control_pts_less_than_3 = np.array([[0, 0], [0, 0]])
    with pytest.raises(ValueError):
        _ensure_2d(control_pts_less_than_3)


def test_tps_transform_estimation():
    tform = TPSTransform()
    assert tform.estimate(DST, SRC)
    assert tform._estimated is True
    assert len(tform.control_points) > 0

    assert len(tform.parameters) > 0
    assert tform.parameters.shape[0] ==  SRC.shape[0] + 3
    np.testing.assert_array_equal(tform.control_points, SRC)

def test_tps_transform_init():
    tps = TPSTransform()

    # Test that _estimated is initialized to False
    assert tps._estimated is False

    # Test that parameters is an empty array of dtype float32
    assert isinstance(tps.parameters, np.ndarray)
    assert tps.parameters.dtype == np.float32
    assert tps.parameters.size == 0

    # Test that control_points is an empty array of dtype float32
    assert isinstance(tps.control_points, np.ndarray)
    assert tps.control_points.dtype == np.float32
    assert tps.control_points.size == 0


def test_warp_tform():
    image = ski.data.checkerboard()
    tform = TPSTransform()
    assert hasattr(tform, 'transform')

    # Test warp before estimate is computed
    with pytest.raises(ValueError):
        tps_warp(image, tform)

    # Test warp after estimate is computed
    tform.estimate(SRC, DST)
    output = tps_warp(image, tform)

    assert image.shape == output.shape

def test_tps_warp_resizing():
    pass

def test_tps_warp_rotation():
    pass

def test_tps_warp_translation():
    pass


def test_tps_transform_call():
    pass
