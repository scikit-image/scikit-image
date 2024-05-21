import numpy as np
import pytest

from skimage.future import ThinPlateSplineTransform

SRC = np.array([[0, 0], [0, 5], [5, 5], [5, 0]])

DST = np.array([[5, 0], [0, 0], [0, 5], [5, 5]])


class TestThinPlateSplineTransform:
    def test_tps_transform_init(self):
        tform = ThinPlateSplineTransform()

        # Test that _estimated is initialized to False
        assert tform._estimated is False
        assert tform._spline_mappings is None
        assert tform.src is None

    def test_tps_transform_inverse(self):
        tps = ThinPlateSplineTransform()
        with pytest.raises(NotImplementedError):
            tps.inverse()

    def test_tps_transform_estimation(self):
        src = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
        dst = np.array([[0, 0.75], [-1, 0.25], [0, -1.25], [1, 0.25]], dtype=np.float32)
        tform = ThinPlateSplineTransform()
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
        assert tform._spline_mappings is None
        assert tform.src is None

        # Perform estimation
        assert tform.estimate(src, dst) is True
        np.testing.assert_array_equal(tform.src, src)

        assert tform._spline_mappings.shape == (src.shape[0] + 3, 2)

        np.testing.assert_allclose(
            tform._spline_mappings,
            desired_spline_mappings,
            rtol=0.1,
            atol=1e-16,
        )

    def test_tps_transform_estimation_failure(self):
        # Test the estimate method when the estimation fails
        tform = ThinPlateSplineTransform()
        src = np.array([[0, 0], [0, 5], [5, 5], [5, 0]])
        dst = np.array([[5, 0], [0, 0], [0, 5]])

        # Ensure that the initial state is as expected
        assert tform._estimated is False
        assert tform._spline_mappings is None
        assert tform.src is None

        # Perform the estimation, which should fail due to the mismatched number of points
        with pytest.raises(ValueError, match=".*coordinates must match"):
            tform.estimate(src, dst)

        # # Check if src and dst have fewer than 3 points
        with pytest.raises(
            ValueError, match="There should be at least 3 points in both sets*."
        ):
            src_less_than_3pts = np.array([[0, 0], [0, 5]])
            tform.estimate(src_less_than_3pts, dst)

            dst_less_than_3pts = np.array([[0, 0], [0, 5]])
            tform.estimate(src, dst_less_than_3pts)

            tform.estimate(src_less_than_3pts, dst_less_than_3pts)

        # Check that src or dst not being (N, 2) does error
        with pytest.raises(ValueError):
            src_not_2d = np.array([0, 1, 2, 3])
            tform.estimate(src_not_2d, dst)

            dst_not_2d = np.array([[1, 2, 3], [4, 5, 6]])
            tform.estimate(src, dst_not_2d)

        # Check that, when the estimation fails, the instance attributes remain unchanged
        assert tform._estimated is False
        assert tform._spline_mappings is None
        assert tform.src is None
