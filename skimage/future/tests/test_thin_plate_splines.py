import numpy as np
import pytest
from scipy.spatial import distance_matrix

from skimage.future import ThinPlateSplineTransform

SRC = np.array([[0, 0], [0, 5], [5, 5], [5, 0]])

DST = np.array([[5, 0], [0, 0], [0, 5], [5, 5]])


class TestThinPlateSplineTransform:
    def test_tps_transform_init(self):
        tps = ThinPlateSplineTransform()

        # Test that _estimated is initialized to False
        assert tps._estimated is False
        assert tps._spline_mappings is None
        assert tps.src is None

    def test_call_before_estimation(self):
        tps = ThinPlateSplineTransform()
        coords = np.array([[0, 0], [0, 5]])
        with pytest.raises(ValueError, match="Transformation is undefined"):
            tps(coords)

    def test_call_invalid_coords_shape(self):
        tps = ThinPlateSplineTransform()
        coords = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            tps(coords)

    def test_call_valid_transformation(self):
        tps = ThinPlateSplineTransform()
        tps.estimate(SRC, DST)
        coords = np.array([[2.5, 2.5], [1, 1]])
        result = tps(coords)
        assert result.shape == (2, 2)

    def test_tps_transform_inverse(self):
        tps = ThinPlateSplineTransform()
        with pytest.raises(NotImplementedError):
            tps.inverse()

    def test_radial_distance_basic(self):
        tps = ThinPlateSplineTransform()
        src = np.array([[0, 0], [0, 5], [5, 5]])
        tps.src = src
        coords = np.array([[0, 0], [0, 5]])
        expected_dists = distance_matrix(coords, src)
        expected_kernel = tps._radial_basis_kernel(expected_dists)
        result = tps._radial_distance(coords)
        np.testing.assert_array_equal(result, expected_kernel)

    def test_radial_basis_small_values(self):
        r = np.array([[1e-10, 1e-8], [1e-6, 1e-4]])
        expected = np.array([[0.0, 0.0], [0.0, 0.0]])
        result = ThinPlateSplineTransform._radial_basis_kernel(r)
        np.testing.assert_almost_equal(result, expected, decimal=6)

    def test_tps_transform_estimation(self):
        src = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
        dst = np.array([[0, 0.75], [-1, 0.25], [0, -1.25], [1, 0.25]], dtype=np.float32)
        tps = ThinPlateSplineTransform()
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
        assert tps._spline_mappings is None
        assert tps.src is None

        # Perform estimation
        assert tps.estimate(src, dst) is True
        np.testing.assert_array_equal(tps.src, src)

        assert tps._spline_mappings.shape == (src.shape[0] + 3, 2)

        np.testing.assert_allclose(
            tps._spline_mappings,
            desired_spline_mappings,
            rtol=0.1,
            atol=1e-16,
        )

    def test_tps_transform_estimation_failure(self):
        # Test the estimate method when the estimation fails
        tps = ThinPlateSplineTransform()
        src = np.array([[0, 0], [0, 5], [5, 5], [5, 0]])
        dst = np.array([[5, 0], [0, 0], [0, 5]])

        # Ensure that the initial state is as expected
        assert tps._estimated is False
        assert tps._spline_mappings is None
        assert tps.src is None

        # Perform the estimation, which should fail due to the mismatched number of points
        with pytest.raises(ValueError, match="Shape of `src` and `dst` didn't match"):
            tps.estimate(src, dst)

        # # Check if src and dst have fewer than 3 points
        with pytest.raises(ValueError, match="Need at least 3 points"):
            src_less_than_3pts = np.array([[0, 0], [0, 5]])
            tps.estimate(src_less_than_3pts, dst)

            dst_less_than_3pts = np.array([[0, 0], [0, 5]])
            tps.estimate(src, dst_less_than_3pts)

            tps.estimate(src_less_than_3pts, dst_less_than_3pts)

        # Check that src or dst not being (N, 2) does error
        with pytest.raises(ValueError):
            src_not_2d = np.array([0, 1, 2, 3])
            tps.estimate(src_not_2d, dst)

            dst_not_2d = np.array([[1, 2, 3], [4, 5, 6]])
            tps.estimate(src, dst_not_2d)

        # Check that, when the estimation fails, the instance attributes remain unchanged
        assert tps._estimated is False
        assert tps._spline_mappings is None
        assert tps.src is None

    def test_estimate_spline_mappings(self):
        tps = ThinPlateSplineTransform()
        tps.estimate(SRC, DST)
        assert tps._spline_mappings is not None
        assert tps._spline_mappings.shape == (len(SRC) + 3, 2)
