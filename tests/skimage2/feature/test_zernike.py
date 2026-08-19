import re
import pytest
import numpy as np
from _skimage2.feature import zernike_features
from _skimage2.feature.zernike import ZernikeResults


@pytest.fixture
def sample_image():
    """Create a dummy image for tests."""
    img = np.zeros((127, 127))
    img[38:89, 38:89] = 255
    return img


class TestZernikeFeatures:
    """Tests for normal/regular usage of the ZF API."""

    def test_default_zernike_features(self, sample_image):
        """Test for default ZFs with degree, radius, center given by user."""
        feature_type = "default"
        degree = 5
        radius = 38
        center = np.array([63, 63])
        return_complex_moments = False
        results = ZernikeResults(
            features=np.array(
                [
                    5.76335032e-01,
                    1.55789700e-17,
                    6.91282739e-01,
                    3.30196655e-17,
                    4.34968136e-17,
                    3.26815381e-17,
                    1.27767584e-01,
                    4.75458231e-16,
                    1.55826037e-01,
                    4.13079265e-17,
                    3.33720848e-17,
                    2.09353610e-17,
                ],
                dtype=np.float64,
            ),
            complex_moments=None,
            radius=38.0,
            center_coord=np.array([63.0, 63.0], dtype=np.float64),
        )
        znres = zernike_features(
            image=sample_image,
            feature_type=feature_type,
            degree=degree,
            radius=radius,
            center_coord=center,
            return_complex_moments=return_complex_moments,
        )
        assert isinstance(results, ZernikeResults)
        assert isinstance(znres, ZernikeResults)
        assert np.all(znres.features) == np.all(results.features)
        assert znres.complex_moments == results.complex_moments
        assert znres.radius == results.radius
        assert np.all(znres.center_coord) == np.all(results.center_coord)

    def test_pseudo_zernike_features(self, sample_image):
        """Test for pseudo ZFs with degree given, but radius, center are auto."""
        feature_type = "pseudo"
        degree = 5
        radius = "auto"
        center = "auto"
        return_complex_moments = True
        results = ZernikeResults(
            features=np.array(
                [
                    3.47782629e-01,
                    1.03675405e-01,
                    1.86882747e-17,
                    3.06387237e-01,
                    2.27686095e-17,
                    1.26912214e-16,
                    6.42298357e-02,
                    6.51737232e-18,
                    1.60217837e-17,
                    2.23480834e-17,
                    9.78544850e-02,
                    2.08472082e-17,
                    1.46213712e-16,
                    2.90064471e-17,
                    1.81683376e-01,
                    9.62118378e-02,
                    1.09049136e-17,
                    3.76528544e-17,
                    3.17981638e-17,
                    1.44028202e-01,
                    6.77181551e-17,
                ]
            ),
            complex_moments=np.array(
                [
                    3.47782629e-01 + 0.00000000e00j,
                    -1.03675405e-01 + 0.00000000e00j,
                    -4.93067220e-19 - 1.86817691e-17j,
                    -3.06387237e-01 + 0.00000000e00j,
                    8.29996488e-18 - 2.12018905e-17j,
                    -1.26803190e-16 + 5.25938368e-18j,
                    -6.42298357e-02 + 0.00000000e00j,
                    -2.95840332e-18 + 5.80723615e-18j,
                    1.48452124e-17 + 6.02637714e-18j,
                    9.64220342e-18 - 2.01609708e-17j,
                    9.78544850e-02 + 0.00000000e00j,
                    1.09570493e-18 + 2.08183938e-17j,
                    1.46212108e-16 + 6.84815584e-19j,
                    1.83530576e-17 + 2.24619512e-17j,
                    -1.81683376e-01 + 9.79149150e-17j,
                    9.62118378e-02 + 0.00000000e00j,
                    -2.87622545e-18 - 1.05187674e-17j,
                    -3.65919129e-17 - 8.87520997e-18j,
                    1.10118346e-17 + 2.98305668e-17j,
                    1.44028202e-01 - 8.78218337e-17j,
                    -4.76631646e-18 - 6.75502092e-17j,
                ]
            ),
            radius=36.0,
            center_coord=np.array([63.0, 63.0]),
        )

        znres = zernike_features(
            image=sample_image,
            feature_type=feature_type,
            degree=degree,
            radius=radius,
            center_coord=center,
            return_complex_moments=return_complex_moments,
        )
        assert isinstance(results, ZernikeResults)
        assert isinstance(znres, ZernikeResults)
        assert np.all(znres.features) == np.all(results.features)
        assert np.all(znres.complex_moments) == np.all(results.complex_moments)
        assert znres.radius == results.radius
        assert np.all(znres.center_coord) == np.all(results.center_coord)

    @pytest.mark.parametrize(
        "one_dtype", [np.uint8, np.float16, np.float32, np.float64]
    )
    def test_image_types_default_zernike_features(self, sample_image, one_dtype):
        """Test for various image datatypes for default ZFs."""
        feature_type = "default"
        degree = 5
        radius = 38
        center = np.array([63, 63])
        return_complex_moments = False
        results = ZernikeResults(
            features=np.array(
                [
                    5.76335032e-01,
                    1.55789700e-17,
                    6.91282739e-01,
                    3.30196655e-17,
                    4.34968136e-17,
                    3.26815381e-17,
                    1.27767584e-01,
                    4.75458231e-16,
                    1.55826037e-01,
                    4.13079265e-17,
                    3.33720848e-17,
                    2.09353610e-17,
                ],
                dtype=np.float64,
            ),
            complex_moments=None,
            radius=38.0,
            center_coord=np.array([63.0, 63.0], dtype=np.float64),
        )
        znres = zernike_features(
            image=sample_image.astype(one_dtype),
            feature_type=feature_type,
            degree=degree,
            radius=radius,
            center_coord=center,
            return_complex_moments=return_complex_moments,
        )
        assert isinstance(results, ZernikeResults)
        assert isinstance(znres, ZernikeResults)
        assert np.all(znres.features) == np.all(results.features)
        assert znres.complex_moments == results.complex_moments
        assert znres.radius == results.radius
        assert np.all(znres.center_coord) == np.all(results.center_coord)


class TestZFErrors:
    """Test various errors thrown by the API."""

    @pytest.mark.parametrize(
        "bad_image",
        [8, None, [], (0, 55, 1), {"abc": None}, True, False],
    )
    def test_image_typeerrors(self, bad_image):
        """Test image related type errors."""
        err_str = "'image' must be a numpy array."
        feature_type = "default"
        degree = 5
        radius = 38
        center = np.array([63, 63])
        return_complex_moments = False
        with pytest.raises(TypeError, match=err_str):
            zernike_features(
                image=bad_image,
                feature_type=feature_type,
                degree=degree,
                radius=radius,
                center_coord=center,
                return_complex_moments=return_complex_moments,
            )

    def test_image_valuerrors(self, sample_image):
        """Test image related value errors."""
        err_str = "Currently, only single channel images are supported."
        feature_type = "default"
        degree = 5
        radius = 38
        center = np.array([63, 63])
        return_complex_moments = False
        with pytest.raises(ValueError, match=err_str):
            zernike_features(
                image=sample_image.flatten(),
                feature_type=feature_type,
                degree=degree,
                radius=radius,
                center_coord=center,
                return_complex_moments=return_complex_moments,
            )
        with pytest.raises(ValueError, match=err_str):
            zernike_features(
                image=sample_image[:, :, np.newaxis],
                feature_type=feature_type,
                degree=degree,
                radius=radius,
                center_coord=center,
                return_complex_moments=return_complex_moments,
            )

    @pytest.mark.parametrize(
        "bad_feature_type",
        [8, np.array([10.0, 11]), None, [], (0, 55, 1), {"abc": None}, True, False],
    )
    def test_feature_type_typeerrors(self, sample_image, bad_feature_type):
        """Test feature_type related type errors."""
        err_str = "'feature_type' can only be string."
        feature_type = bad_feature_type
        degree = 5
        radius = 38
        center = np.array([63, 63])
        return_complex_moments = False
        with pytest.raises(TypeError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                degree=degree,
                radius=radius,
                center_coord=center,
                return_complex_moments=return_complex_moments,
            )

    @pytest.mark.parametrize(
        "bad_feature_type",
        ["dasda", "defautl", "dens", "PSEUDO", "DefAULT", "densE"],
    )
    def test_feature_type_valueerrors(self, sample_image, bad_feature_type):
        """Test feature_type related value errors."""
        err_str = "'feature_type' is not a valid choice. Choose 'default', or 'dense'/'pseudo' for denser pseudo-Zernike features."
        feature_type = bad_feature_type
        degree = 5
        radius = 38
        center = np.array([63, 63])
        return_complex_moments = False
        with pytest.raises(ValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                degree=degree,
                radius=radius,
                center_coord=center,
                return_complex_moments=return_complex_moments,
            )

    @pytest.mark.parametrize(
        "bad_degree",
        [8.0, np.array([10.0, 11]), None, [], (0, 55, 1), {"abc": None}],
    )
    def test_degree_typeerrors(self, sample_image, bad_degree):
        """Test degree related type errors."""
        err_str = "'degree' can only be interger value."
        feature_type = "default"
        degree = bad_degree
        radius = 38
        center = np.array([63, 63])
        return_complex_moments = False
        with pytest.raises(TypeError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                degree=degree,
                radius=radius,
                center_coord=center,
                return_complex_moments=return_complex_moments,
            )

    @pytest.mark.parametrize(
        "bad_degree",
        [False, -99, -1, 0],
    )
    def test_degree_valueerrors(self, sample_image, bad_degree):
        """Test degree related value errors."""
        err_str = "'degree' is not a valid positive integer value."
        feature_type = "default"
        degree = bad_degree
        radius = 38
        center = np.array([63, 63])
        return_complex_moments = False
        with pytest.raises(ValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                degree=degree,
                radius=radius,
                center_coord=center,
                return_complex_moments=return_complex_moments,
            )

    @pytest.mark.parametrize(
        "bad_degree",
        [38, 100],
    )
    def test_degree_runtimewarning(self, sample_image, bad_degree):
        """Test degree related runtime warnings."""
        err_str = "'degree' is a large value compared to 'radius'. Feature computations might be unstable or slow."
        feature_type = "default"
        degree = bad_degree
        radius = 38
        center = np.array([63, 63])
        return_complex_moments = False
        with pytest.warns(RuntimeWarning, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                degree=degree,
                radius=radius,
                center_coord=center,
                return_complex_moments=return_complex_moments,
            )

    @pytest.mark.parametrize(
        "bad_radius",
        [np.array([10.0, 11]), None, [], (0, 55, 1), {"abc": None}],
    )
    def test_radius_typeerrors(self, sample_image, bad_radius):
        """Test radius related type errors."""
        err_str = "'radius' is not a correct choice or datatype."
        feature_type = "default"
        degree = 5
        radius = bad_radius
        center = np.array([63, 63])
        return_complex_moments = False
        with pytest.raises(TypeError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                degree=degree,
                radius=radius,
                center_coord=center,
                return_complex_moments=return_complex_moments,
            )

    @pytest.mark.parametrize(
        "bad_radius",
        [-99, -1, -0.0, 0.0, 1.0, 128, "AUTO", "autO", "].';.", "", 70],
    )
    def test_radius_valueerrors(self, sample_image, bad_radius):
        """Test radius related value errors."""
        err_str1 = re.escape("'radius' value is invalid for image size.")
        err_str2 = re.escape("'center_coord' and 'radius' exceed image size.")
        err_str3 = re.escape("'radius' only supports 'auto' as string choice.")
        err_str = f"{err_str1}|{err_str2}|{err_str3}"
        feature_type = "default"
        degree = 5
        radius = bad_radius
        center = np.array([63, 63])
        return_complex_moments = False
        with pytest.raises(ValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                degree=degree,
                radius=radius,
                center_coord=center,
                return_complex_moments=return_complex_moments,
            )

    @pytest.mark.parametrize(
        "bad_center",
        [8.0, False, None, [], (0, 55, 1), {"abc": None}],
    )
    def test_center_typeerrors(self, sample_image, bad_center):
        """Test center related type errors."""
        err_str = "'center_coord' is not a valid choice."
        feature_type = "default"
        degree = 5
        radius = 38
        center = bad_center
        return_complex_moments = False
        with pytest.raises(TypeError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                degree=degree,
                radius=radius,
                center_coord=center,
                return_complex_moments=return_complex_moments,
            )

    @pytest.mark.parametrize(
        "bad_center",
        [
            np.array([[1], [2]]),
            np.array([1, 2, 3, 4]),
            "agadf",
            "AUTO",
            "autO",
            "",
            np.array([0, 0]),
            np.array([125, 0]),
            np.array([0, 125]),
            np.array([125, 125]),
        ],
    )
    def test_center_valueerrors(self, sample_image, bad_center):
        """Test center related value errors."""
        err_str1 = re.escape("'center_coord' expects 1D array of shape '(2,)'.")
        err_str2 = re.escape("'center_coord' and 'radius' exceed image size.")
        err_str3 = re.escape("'center_coord' only supports 'auto' as string choice.")
        err_str = f"{err_str1}|{err_str2}|{err_str3}"
        feature_type = "default"
        degree = 5
        radius = 38
        center = bad_center
        return_complex_moments = False
        with pytest.raises(ValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                degree=degree,
                radius=radius,
                center_coord=center,
                return_complex_moments=return_complex_moments,
            )

    @pytest.mark.parametrize(
        "bad_radius, bad_center",
        [
            ("auto", np.array([63, 63])),
            (38, "auto"),
        ],
    )
    def test_radius_center_combined_auto_valueerror(
        self, sample_image, bad_radius, bad_center
    ):
        """Test radius, center related combined value errors."""
        err_str = re.escape(
            "Both 'radius' and 'center_coord' need to be 'auto' for automated calculations."
        )
        feature_type = "default"
        degree = 5
        radius = bad_radius
        center = bad_center
        return_complex_moments = False
        with pytest.raises(ValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                degree=degree,
                radius=radius,
                center_coord=center,
                return_complex_moments=return_complex_moments,
            )
