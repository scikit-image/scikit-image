import numpy as np
import pytest
from numpy.testing import assert_array_equal

from skimage.transform._thin_plate_splines import TpsTransform, _ensure_2d, tps_warp

SRC = np.array([[0, 0], [0, 5], [5, 5], [5, 0]])

DST = np.array([[5, 0], [0, 0], [0, 5], [5, 5]])


class TestTpsTransform:
    def test_tps_transform_init(self):
        tform = TpsTransform()

        # Test that _estimated is initialized to False
        assert tform._estimated is False
        assert tform.coefficients is None
        assert tform.src is None

    def test_tps_transform_inverse(self):
        tps = TpsTransform()
        with pytest.raises(NotImplementedError):
            tps.inverse()

    def test_tps_transform_ensure_2d(self):
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

    def test_tps_transform_estimation(self):
        src = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
        dst = np.array([[0, 0.75], [-1, 0.25], [0, -1.25], [1, 0.25]], dtype=np.float32)
        tform = TpsTransform()

        # Ensure that the initial state is as expected
        assert tform.coefficients is None
        assert tform.src is None

        # Perform estimation
        assert tform.estimate(src, dst) is True
        np.testing.assert_array_equal(tform.src, src)

        assert tform.coefficients.shape == (src.shape[0] + 3, 2)

        np.testing.assert_allclose(
            tform.coefficients[:, 0],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            rtol=0.1,
            atol=1e-16,
        )
        np.testing.assert_allclose(
            tform.coefficients[:, 1],
            np.array([-0.0902, 0.0902, -0.0902, 0.0902, 0.0, 0.0, 1.0]),
            rtol=0.1,
            atol=1e-16,
        )

    def test_tps_transform_estimation_failure(self):
        # Test the estimate method when the estimation fails
        tform = TpsTransform()
        src = np.array([[0, 0], [0, 5], [5, 5], [5, 0]])
        dst = np.array([[5, 0], [0, 0], [0, 5]])

        # Ensure that the initial state is as expected
        assert tform._estimated is False
        assert tform.coefficients is None
        assert tform.src is None

        # Perform the estimation, which should fail due to the mismatched number of points
        with pytest.raises(ValueError, match=".*shape must be identical"):
            tform.estimate(src, dst)

        # Check if the estimation failed and the instance attributes remain unchanged
        assert tform._estimated is False
        assert tform.coefficients is None
        assert tform.src is None


class TestTpsWarp:
    @pytest.mark.parametrize("image_shape", [0, (0, 10), (10, 0)])
    def test_tps_warp_invalid_image_shape(self, image_shape):
        img = np.zeros(image_shape)

        with pytest.raises(
            ValueError, match="Cannot warp empty image with dimensions."
        ):
            tps_warp(img, SRC, DST)
        with pytest.raises(
            ValueError, match="Cannot warp empty image with dimensions."
        ):
            tps_warp(img, SRC, DST)
        with pytest.raises(
            ValueError, match="Cannot warp empty image with dimensions."
        ):
            tps_warp(img, SRC, DST)

    def test_tps_warp_valid_output_region(self):
        img = np.zeros((100, 100))
        valid_output_region = (0, 0, 100, 100)
        result = tps_warp(img, SRC, DST, output_region=valid_output_region)
        assert result.shape == img.shape

    @pytest.mark.parametrize("invalid_output_region", [0, (0,), (10, 10), (10, 10, 10)])
    def test_invalid_invalid_output_region(self, invalid_output_region):
        img = np.zeros((100, 100))

        with pytest.raises(ValueError):
            tps_warp(img, SRC, DST, output_region=invalid_output_region)

    def test_tps_transform_call(self):
        # Test __call__ method without esitmate
        tform = TpsTransform()
        # Define coordinates to transform using meshgrid
        coords = np.array(np.mgrid[0:5, 0:5])
        coords = coords.T.reshape(-1, 2)

        # Call a TpsTransform without estimate
        with pytest.raises(ValueError, match="None. Compute the `estimate`"):
            tform(coords)

        # Test __call__ method with estimmate
        tform.estimate(SRC, DST)
        trans_coord = tform(coords)
        yy_trans = trans_coord[:, 1]

        # fmt: off
        expected_yy = np.array([0, 1.0, 2.0, 3.0, 4.0,
                                0, 1.0, 2.0, 3.0, 4.0,
                                0, 1.0, 2.0, 3.0, 4.0,
                                0, 1.0, 2.0, 3.0, 4.0,
                                0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        # fmt: on
        np.testing.assert_allclose(yy_trans, expected_yy)

    def test_tps_warp_resizing(self):
        pass

    def test_tps_warp_rotation(self):
        pass

    def test_tps_warp_translation(self):
        pass
