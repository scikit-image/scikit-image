import numpy as np
import pytest

import skimage as ski
from skimage.transform import ThinPlateSplineTransform

SRC = np.array([[0, 0], [0, 5], [5, 5], [5, 0]])

DST = np.array([[5, 0], [0, 0], [0, 5], [5, 5]])


class TestThinPlateSplineTransform:
    tform_class = ThinPlateSplineTransform

    def test_call_before_estimation(self):
        tps = self.tform_class()
        assert tps.src is None
        with pytest.raises(ValueError, match="Transformation is undefined"):
            tps(SRC)

    def test_call_invalid_coords_shape(self):
        tps = self.tform_class.from_estimate(SRC, DST)
        coords = np.array([1, 2, 3])
        with pytest.raises(
            ValueError, match=r"Input `coords` must have shape \(N, 2\)"
        ):
            tps(coords)

    def test_call_on_SRC(self):
        tps = self.tform_class.from_estimate(SRC, DST)
        result = tps(SRC)
        np.testing.assert_allclose(result, DST, atol=1e-15)

    def test_tps_transform_inverse(self):
        tps = self.tform_class.from_estimate(SRC, DST)
        with pytest.raises(NotImplementedError):
            tps.inverse()

    def test_tps_estimation_faulty_input(self):
        src = np.array([[0, 0], [0, 5], [5, 5], [5, 0]])
        dst = np.array([[5, 0], [0, 0], [0, 5]])

        with pytest.raises(ValueError, match="Shape of `src` and `dst` didn't match"):
            tps = self.tform_class.from_estimate(src, dst)

        less_than_3pts = np.array([[0, 0], [0, 5]])
        with pytest.raises(ValueError, match="Need at least 3 points"):
            self.tform_class.from_estimate(less_than_3pts, dst)
        with pytest.raises(ValueError, match="Need at least 3 points"):
            self.tform_class.from_estimate(src, less_than_3pts)
        with pytest.raises(ValueError, match="Need at least 3 points"):
            self.tform_class.from_estimate(less_than_3pts, less_than_3pts)

        not_2d = np.array([0, 1, 2, 3])
        with pytest.raises(ValueError, match=".*`src` must be a 2-dimensional array"):
            self.tform_class.from_estimate(not_2d, dst)
        with pytest.raises(ValueError, match=".*`dst` must be a 2-dimensional array"):
            self.tform_class.from_estimate(src, not_2d)

        # When the estimation fails, the instance attributes remain unchanged
        tps = self.tform_class()
        assert tps.src is None
        with pytest.warns(FutureWarning, match='`estimate` is deprecated'):
            with pytest.raises(
                ValueError, match=".*`dst` must be a 2-dimensional array"
            ):
                tps.estimate(src, not_2d)
        assert tps.src is None

    def test_rotate(self):
        image = ski.data.astronaut()
        desired = ski.transform.rotate(image, angle=90)

        src = np.array([[0, 0], [0, 511], [511, 511], [511, 0]])
        dst = np.array([[511, 0], [0, 0], [0, 511], [511, 511]])
        tps = self.tform_class.from_estimate(src, dst)
        result = ski.transform.warp(image, tps)

        np.testing.assert_allclose(result, desired, atol=1e-13)

        # Estimate method.
        tps2 = self.tform_class()
        with pytest.warns(FutureWarning, match='`estimate` is deprecated'):
            assert tps2.estimate(src, dst)
        result = ski.transform.warp(image, tps2)
        np.testing.assert_allclose(result, desired, atol=1e-13)
