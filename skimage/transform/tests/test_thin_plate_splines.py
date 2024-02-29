import numpy as np
import pytest

import skimage as ski
from skimage.transform._thin_plate_splines import TpsTransform, tps_warp

SRC = np.array([[0, 0], [0, 5], [5, 5], [5, 0]])

DST = np.array([[5, 0], [0, 0], [0, 5], [5, 5]])


class TestTpsTransform:
    def test_tps_transform_init(self):
        tform = TpsTransform()

        # Test that _estimated is initialized to False
        assert tform._estimated is False
        assert tform.spline_mappings is None
        assert tform.src is None

    def test_tps_transform_inverse(self):
        tps = TpsTransform()
        with pytest.raises(NotImplementedError):
            tps.inverse()

    def test_tps_transform_estimation(self):
        src = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
        dst = np.array([[0, 0.75], [-1, 0.25], [0, -1.25], [1, 0.25]], dtype=np.float32)
        tform = TpsTransform()
        desired_spline_mappings = np.array(
            [
                [0.0, -0.0902],
                [0.0, 0.0902],
                [0.0, -0.0902],
                [0.0, 0.0902],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        # Ensure that the initial state is as expected
        assert tform.spline_mappings is None
        assert tform.src is None

        # Perform estimation
        assert tform.estimate(src, dst) is True
        np.testing.assert_array_equal(tform.src, src)

        assert tform.spline_mappings.shape == (src.shape[0] + 3, 2)

        np.testing.assert_allclose(
            tform.spline_mappings,
            desired_spline_mappings,
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
        assert tform.spline_mappings is None
        assert tform.src is None

        # Perform the estimation, which should fail due to the mismatched number of points
        with pytest.raises(ValueError, match=".*coordinates must match"):
            tform.estimate(src, dst)

        # # Check if src and dst have fewer than 3 points
        with pytest.raises(
            ValueError, match=".*points less than 3 is considered undefined."
        ):
            src_less_than_3pts = np.array([[0, 0], [0, 5]])
            tform.estimate(src_less_than_3pts, dst)

            dst_less_than_3pts = np.array([[0, 0], [0, 5]])
            tform.estimate(src, dst_less_than_3pts)

            tform.estimate(src_less_than_3pts, dst_less_than_3pts)

        # Check if src or dst not being (N, 2)
        with pytest.raises(ValueError):
            src_not_2d = np.array([0, 1, 2, 3])
            tform.estimate(src_not_2d, dst)

            dst_not_2d = np.array([[1, 2, 3], [4, 5, 6]])
            tform.estimate(src, dst_not_2d)

        # Check if the estimation failed and the instance attributes remain unchanged
        assert tform._estimated is False
        assert tform.spline_mappings is None
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

    @pytest.mark.parametrize("image_array", [(2, 2, 2, 2), (4, 4, 4, 4)])
    def test_tps_warp_invalid_image_dimension(self, image_array):
        with pytest.raises(ValueError, match="Only 2D and 3D images are supported"):
            tps_warp(image_array, SRC, DST)

    @pytest.mark.parametrize("invalid_grid_scaling", [0, -10])
    def test_invalid_grid_scaling(self, invalid_grid_scaling):
        img = np.zeros((100, 100))
        with pytest.raises(
            ValueError, match="Grid scaling must be equal to or greater than 1."
        ):
            tps_warp(img, SRC, DST, grid_scaling=invalid_grid_scaling)

    @pytest.mark.parametrize("valid_grid_scaling", [1, 2, 7])
    def test_valid_grid_scaling(self, valid_grid_scaling):
        img = np.zeros((100, 100))

        result = tps_warp(img, SRC, DST, grid_scaling=valid_grid_scaling)
        assert result.shape == img.shape

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

    @pytest.mark.parametrize(
        "invalid_output_region", [(0, 0, 10, 0), (0, 0, 0, 10), (1, 1, 0, 10)]
    )
    def test_output_region(self, invalid_output_region):
        img = np.zeros((100, 100))
        # Test case where x_steps and y_steps are both zero
        with pytest.raises(RuntimeError):
            tps_warp(img, SRC, DST, output_region=invalid_output_region)

    @pytest.mark.parametrize("valid_grid_scaling", [1, 2, 7])
    def test_rgb_image_channels(self, valid_grid_scaling):
        rgb_img = np.zeros((100, 100, 3), dtype=np.uint8)

        rgb_img[:, :, 0] = 255
        rgb_img[:, :, 1] = 0
        rgb_img[:, :, 2] = 0

        result = tps_warp(rgb_img, SRC, DST, grid_scaling=valid_grid_scaling)
        assert result.shape[-1] == rgb_img.shape[-1]
        assert np.min(result) >= 0
        assert np.max(result) <= 255

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

    def test_zoom_in(self):
        # fmt: off
        image = np.array(
            [[9, 9, 9, 9, 9, 9, 9],
             [9, 0, 0, 0, 0, 0, 9],
             [9, 0, 1, 1, 1, 0, 9],
             [9, 0, 1, 2, 1, 0, 9],
             [9, 0, 1, 1, 1, 0, 9],
             [9, 0, 0, 0, 0, 0, 9],
             [9, 9, 9, 9, 9, 9, 9]],
            dtype=float,
        )
        desired = np.array(
            [[0.25, 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.25],
             [0.5 , 1.  , 1.  , 1.  , 1.  , 1.  , 0.5 ],
             [0.5 , 1.  , 1.25, 1.5 , 1.25, 1.  , 0.5 ],
             [0.5 , 1.  , 1.5 , 2.  , 1.5 , 1.  , 0.5 ],
             [0.5 , 1.  , 1.25, 1.5 , 1.25, 1.  , 0.5 ],
             [0.5 , 1.  , 1.  , 1.  , 1.  , 1.  , 0.5 ],
             [0.25, 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.25]]
        )
        # fmt: on
        src = np.array([[2, 2], [2, 4], [4, 4], [4, 2]])
        dst = np.array([[1, 1], [1, 5], [5, 5], [5, 1]])
        result = tps_warp(image, src=src, dst=dst)
        np.testing.assert_array_equal(result, desired)

    @pytest.mark.parametrize("quadrant_shift", [1, -1, 2, -2, 3, -3, 4, -4])
    def test_rotate(self, quadrant_shift):
        image = np.linspace(1, 9, num=9).reshape((3, 3))
        desired = ski.transform.rotate(image, angle=90 * quadrant_shift)
        src = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
        dst = np.roll(src, shift=quadrant_shift, axis=0)
        result = tps_warp(image, src=src, dst=dst)
        np.testing.assert_allclose(result, desired)

    def test_tps_warp_translation(self):
        image = np.zeros((5, 5))
        image[:2, :2] = 1
        desired = np.zeros((5, 5))
        desired[2:4, 2:4] = 1
        src = np.array([[1, 1], [1, 2], [2, 2], [2, 1]])
        dst = src + 2
        result = tps_warp(image, src=src, dst=dst)
        np.testing.assert_array_equal(result, desired)
