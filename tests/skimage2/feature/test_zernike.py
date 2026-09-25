import re
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_array_less
from _skimage2.feature import zernike_features
from _skimage2.feature.zernike import (
    ZernikeResults,
    ZernikeTypeError,
    ZernikeValueError,
)


def build_valid_combos():
    valid_pairs = (
        (('circle', 38, None, np.array([60, 60])), (38.0, 0.0, np.array([60.0, 60.0]))),
        (('circle', 'auto', None, 'auto'), (36.0, 0.0, np.array([63.0, 63.0]))),
        (('annulus', 38, 30, np.array([60, 60])), (38.0, 30.0, np.array([60.0, 60.0]))),
        (('annulus', 'auto', 0.2, 'auto'), (36.0, 7.0, np.array([63.0, 63.0]))),
        (('ellipse', 38, 30, np.array([60, 60])), (38.0, 30.0, np.array([60.0, 60.0]))),
        (('ellipse', 30, 38, np.array([60, 60])), (30.0, 38.0, np.array([60.0, 60.0]))),
        (('ellipse', 'auto', None, 'auto'), (29.0, 29.0, np.array([63.0, 63.0]))),
        (
            ('rectangle', 38, 30, np.array([60, 60])),
            (38.0, 30.0, np.array([60.0, 60.0])),
        ),
        (
            ('rectangle', 30, 38, np.array([60, 60])),
            (30.0, 38.0, np.array([60.0, 60.0])),
        ),
        (('rectangle', 'auto', None, 'auto'), (26.0, 26.0, np.array([63.0, 63.0]))),
        (
            ('square', 38, None, np.array([60, 60])),
            (38.0, 38.0, np.array([60.0, 60.0])),
        ),
        (('square', 'auto', None, 'auto'), (26.0, 26.0, np.array([63.0, 63.0]))),
        (
            ('hexagon', 38, None, np.array([60, 60])),
            (38.0, 0.0, np.array([60.0, 60.0])),
        ),
        (('hexagon', 'auto', None, 'auto'), (41.0, 0.0, np.array([63.0, 63.0]))),
    )
    pairs = []
    for odt in (np.uint8, np.float16, np.float32, np.float64):
        for oft in ("conventional", "pseudo"):
            # assumes degree=5, then expect 12 and 21 features resp.
            if oft == "conventional":
                ooshp = (12,)
            else:
                ooshp = (21,)
            for oin, oout in valid_pairs:
                oipt, oipd, oisd, oicc = oin
                oopd, oosd, oocc = oout
                pairs.append(
                    (
                        {
                            "dt": odt,
                            "ft": oft,
                            "pt": oipt,
                            "pd": oipd,
                            "sd": oisd,
                            "cc": oicc,
                        },
                        {"fshp": ooshp, "pd": oopd, "sd": oosd, "cc": oocc},
                    )
                )
    return pairs


@pytest.fixture
def sample_image():
    """Create a dummy image for tests."""
    img = np.zeros((127, 127))
    img[38:89, 38:89] = 255
    return img


class TestZernikeFeatures:
    """Tests for normal/regular usage of the ZF API."""

    def test_conventional_zernike_features(self, sample_image):
        """Test for conventional ZFs with degree, radius, center given by user.

        This is the simplest usage test.
        """
        feature_type = "conventional"
        degree = 5
        pupil_type = "circle"
        primary_dim = 38
        secondary_dim = None
        center_coord = np.array([63, 63])
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        results = ZernikeResults(
            features=np.array(
                [
                    5.76335032e-01,
                    1.25962373e-16,
                    6.91282739e-01,
                    1.62363660e-18,
                    1.25984971e-16,
                    8.36823941e-18,
                    1.27767584e-01,
                    1.11472819e-16,
                    1.55826037e-01,
                    7.57203629e-17,
                    7.72446795e-17,
                    3.59631500e-17,
                ],
                dtype=np.float64,
            ),
            complex_moments=None,
            pupil_mask=None,
            reconstructed_image=None,
            primary_dim=38.0,
            secondary_dim=0.0,
            center_coord=np.array([63.0, 63.0], dtype=np.float64),
        )
        znres = zernike_features(
            image=sample_image,
            feature_type=feature_type,
            pupil_type=pupil_type,
            degree=degree,
            primary_dim=primary_dim,
            secondary_dim=secondary_dim,
            center_coord=center_coord,
            return_complex_moments=return_complex_moments,
            return_pupil_mask=return_pupil_mask,
            return_reconstructed_image=return_reconstructed_image,
        )

        assert isinstance(results, ZernikeResults)
        assert isinstance(znres, ZernikeResults)
        assert_allclose(znres.features, results.features, rtol=1e-6, atol=1e-6)
        assert znres.complex_moments == results.complex_moments
        assert znres.reconstructed_image == results.reconstructed_image
        assert znres.primary_dim == results.primary_dim
        assert znres.secondary_dim == results.secondary_dim
        assert_equal(znres.center_coord, results.center_coord)

    @pytest.mark.parametrize("inc, outc", build_valid_combos())
    def test_zf_valid_combs(self, inc, outc, sample_image):
        """Test for 96 valid combinations of 6 parameters of ZF API.

        The 6 parameters are image datatypes, feature types, pupil types,
        primary_dim, secondary_dim, center_coord.
        """
        feature_type = inc["ft"]
        pupil_type = inc["pt"]
        degree = 5
        primary_dim = inc["pd"]
        secondary_dim = inc["sd"]
        center_coord = inc["cc"]
        return_complex_moments = True
        return_pupil_mask = True
        return_reconstructed_image = True

        znres = zernike_features(
            image=sample_image.astype(inc["dt"]),
            feature_type=feature_type,
            pupil_type=pupil_type,
            degree=degree,
            primary_dim=primary_dim,
            secondary_dim=secondary_dim,
            center_coord=center_coord,
            return_complex_moments=return_complex_moments,
            return_pupil_mask=return_pupil_mask,
            return_reconstructed_image=return_reconstructed_image,
        )

        assert isinstance(znres, ZernikeResults)
        assert isinstance(znres.features, np.ndarray)
        assert znres.features.dtype == np.float64
        assert znres.features.shape == outc["fshp"]
        assert_array_less(0.0, znres.features)
        assert_array_less(znres.features, 1.0)
        assert isinstance(znres.complex_moments, np.ndarray)
        assert znres.complex_moments.dtype == np.complex128
        assert znres.complex_moments.shape == outc["fshp"]
        assert_array_less(-1.0 - 1.0j, znres.complex_moments)
        assert_array_less(znres.complex_moments, 1.0 + 1.0j)
        assert isinstance(znres.pupil_mask, np.ndarray)
        assert znres.pupil_mask.dtype == np.bool_
        assert znres.pupil_mask.shape == sample_image.shape
        assert_equal(np.unique(znres.pupil_mask), np.array([False, True]))
        assert isinstance(znres.reconstructed_image, np.ndarray)
        assert znres.reconstructed_image.dtype == np.uint8
        assert znres.reconstructed_image.shape == sample_image.shape
        assert_array_less(-1, znres.reconstructed_image)
        assert_array_less(znres.reconstructed_image, 256)
        assert isinstance(znres.primary_dim, float)
        assert znres.primary_dim == outc["pd"]
        assert isinstance(znres.secondary_dim, float)
        assert znres.secondary_dim == outc["sd"]
        assert isinstance(znres.center_coord, np.ndarray)
        assert znres.center_coord.dtype == np.float64
        assert znres.center_coord.shape == (2,)
        assert_equal(znres.center_coord, outc["cc"])


class TestZFErrors:
    """Test various errors thrown by the API."""

    @pytest.mark.parametrize(
        "bad_image",
        [8, None, [], (0, 55, 1), {"abc": None}, True, False],
    )
    def test_image_typeerrors(self, bad_image):
        """Test image related type errors."""
        err_str = "'image' must be a numpy array."
        feature_type = "conventional"
        pupil_type = "circle"
        degree = 5
        primary_dim = 38
        secondary_dim = None
        center_coord = np.array([63, 63])
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeTypeError, match=err_str):
            zernike_features(
                image=bad_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    def test_image_valuerrors(self, sample_image):
        """Test image related value errors."""
        err_str = "Only single channel images are supported."
        feature_type = "conventional"
        pupil_type = "circle"
        degree = 5
        primary_dim = 38
        secondary_dim = None
        center_coord = np.array([63, 63])
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeValueError, match=err_str):
            zernike_features(
                image=sample_image.flatten(),
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

        with pytest.raises(ValueError, match=err_str):
            zernike_features(
                image=sample_image[:, :, np.newaxis],
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_feature_type",
        [8, np.array([10.0, 11]), None, [], (0, 55, 1), {"abc": None}, True, False],
    )
    def test_feature_type_typeerrors(self, sample_image, bad_feature_type):
        """Test feature_type related type errors."""
        err_str = "'feature_type' can only be string."
        feature_type = bad_feature_type
        pupil_type = "circle"
        degree = 5
        primary_dim = 38
        secondary_dim = None
        center_coord = np.array([63, 63])
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeTypeError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_feature_type",
        ["';.,$%", "Conventional", "pseud0", "PSEUDO", "DefAULT", "densE"],
    )
    def test_feature_type_valueerrors(self, sample_image, bad_feature_type):
        """Test feature_type related value errors."""
        err_str = (
            "'feature_type' is not a valid choice. Choose 'conventional' or 'pseudo'."
        )
        feature_type = bad_feature_type
        pupil_type = "circle"
        degree = 5
        primary_dim = 38
        secondary_dim = None
        center_coord = np.array([63, 63])
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_degree",
        [8.0, np.array([10.0, 11]), None, [], (0, 55, 1), {"abc": None}],
    )
    def test_degree_typeerrors(self, sample_image, bad_degree):
        """Test degree related type errors."""
        err_str = "'degree' can only be interger value."
        feature_type = "conventional"
        pupil_type = "circle"
        degree = bad_degree
        primary_dim = 38
        secondary_dim = None
        center_coord = np.array([63, 63])
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeTypeError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_degree",
        [False, -99, -1, 0],
    )
    def test_degree_valueerrors(self, sample_image, bad_degree):
        """Test degree related value errors."""
        err_str = "'degree' is not a valid positive integer value."
        feature_type = "conventional"
        pupil_type = "circle"
        degree = bad_degree
        primary_dim = 38
        secondary_dim = None
        center_coord = np.array([63, 63])
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_degree",
        [38, 100],
    )
    def test_degree_runtimewarning(self, sample_image, bad_degree):
        """Test degree related runtime warnings."""
        err_str = "'degree' value is large compared to pupil dimensions. Feature computations might be unstable and slow."
        feature_type = "conventional"
        pupil_type = "circle"
        degree = bad_degree
        primary_dim = 38
        secondary_dim = None
        center_coord = np.array([63, 63])
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.warns(RuntimeWarning, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_pupil_type",
        [False, -99, 100.0, np.array([33, 66, 99]), {"abc": None}, None],
    )
    def test_pupil_type_typeerrors(self, sample_image, bad_pupil_type):
        """Test pupil type related type errors."""
        err_str = "'pupil_type' is not a string."
        feature_type = "conventional"
        pupil_type = bad_pupil_type
        degree = 5
        primary_dim = 38
        secondary_dim = None
        center_coord = np.array([63, 63])
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeTypeError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_pupil_type",
        ["CIRCLE", "1000.0", "hexagone", "$QU@re", "annular", "Ellipse", "reCTaGnLe"],
    )
    def test_pupil_type_valueerrors(self, sample_image, bad_pupil_type):
        """Test pupil type related value errors."""
        err_str = "'pupil_type' is not a valid pupil. Choose one of 6 pupil shapes."
        feature_type = "conventional"
        pupil_type = bad_pupil_type
        degree = 5
        primary_dim = 38
        secondary_dim = None
        center_coord = np.array([63, 63])
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_center_coord",
        [
            False,
            -99,
            100.0,
            np.array([33, 66, 99]),
            np.array([63, 63]),
            {"abc": None},
            None,
        ],
    )
    def test_center_coord_autoerrors(self, sample_image, bad_center_coord):
        """Test center related auto errors."""
        err_str = "When 'primary_dim='auto'', 'center_coord' must be 'auto'."
        feature_type = "conventional"
        pupil_type = "circle"
        degree = 5
        primary_dim = "auto"
        secondary_dim = None
        center_coord = bad_center_coord
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_center_coord",
        [
            np.array([33, 66, 99]),
            np.array([[33], [66], [99]]),
            np.array([[33], [66]]),
            np.zeros((1,)),
            np.array([]),
            np.array([np.nan]),
        ],
    )
    def test_center_coord_arrayerrors(self, sample_image, bad_center_coord):
        """Test center related array errors."""
        err_str = re.escape("'center_coord' must be 1D array of shape (2,).")
        feature_type = "conventional"
        pupil_type = "circle"
        degree = 5
        primary_dim = 38
        secondary_dim = None
        center_coord = bad_center_coord
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_center_coord, bad_primary_dim",
        [
            (np.array([63, 63]), 64),
            (np.array([0, 0]), 38),
            (np.array([127, 0]), 126),
            (np.array([0, 127]), 2),
            (np.array([127, 127]), 20),
            # (np.array([np.nan, np.nan]), 20),
        ],
    )
    def test_center_coord_primary_dim_limiterrors(
        self, sample_image, bad_center_coord, bad_primary_dim
    ):
        """Test center, primary_dim related limits errors."""
        err_str = "'center_coord' and 'primary_dim' exceed image size. Shift the center or reduce primary_dim."
        feature_type = "conventional"
        pupil_type = "circle"
        degree = 5
        primary_dim = bad_primary_dim
        secondary_dim = None
        center_coord = bad_center_coord
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_primary_dim",
        [False, -99, (100.0,), np.array([33, 66, 99]), {"abc": None}, None, []],
    )
    def test_primary_dim_typeerrors(self, sample_image, bad_primary_dim):
        """Test primary_dim related type errors."""
        err_str = "'primary_dim' can only be 'auto' or a number."
        feature_type = "conventional"
        pupil_type = "circle"
        degree = 5
        primary_dim = bad_primary_dim
        secondary_dim = None
        center_coord = np.array([63, 63])
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeTypeError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_secondary_dim",
        [False, -99, (100.0,), np.array([33, 66, 99]), {"abc": None}, []],
    )
    def test_secondary_dim_noneerrors(self, sample_image, bad_secondary_dim):
        """Test secondary_dim related none errors."""
        err_str = "'secondary_dim' is expected to be set to 'None'."
        feature_type = "conventional"
        pupil_type = "circle"
        degree = 5
        primary_dim = 38
        secondary_dim = bad_secondary_dim
        center_coord = np.array([63, 63])
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_secondary_dim",
        [False, True, 0.0, 1.0, 0, 1, -0.0],
    )
    def test_secondary_dim_range01errors(self, sample_image, bad_secondary_dim):
        """Test secondary_dim related (0, 1) errors."""
        err_str = re.escape(
            "For selected pupil, 'secondary_dim' must be in range interval (0.0, 1.0)."
        )
        feature_type = "conventional"
        pupil_type = "annulus"
        degree = 5
        primary_dim = "auto"
        secondary_dim = bad_secondary_dim
        center_coord = "auto"
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_secondary_dim",
        [38, 39, 300, 1e9],
    )
    def test_secondary_dim_rangePDerrors(self, sample_image, bad_secondary_dim):
        """Test secondary_dim annulus related larger than primary_dim errors."""
        err_str = re.escape(
            "For selected pupil, 'secondary_dim' must be in range interval (1.0, 'primary_dim')."
        )
        feature_type = "conventional"
        pupil_type = "annulus"
        degree = 5
        primary_dim = 38
        secondary_dim = bad_secondary_dim
        center_coord = np.array([63, 63])
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )

    @pytest.mark.parametrize(
        "bad_secondary_dim",
        [127, 300, 1e9],
    )
    def test_secondary_dim_limiterrors(self, sample_image, bad_secondary_dim):
        """Test secondary_dim related larger than image errors."""
        err_str = "'secondary_dim' exceed image size. Reduce secondary_dim."
        feature_type = "conventional"
        pupil_type = "rectangle"
        degree = 5
        primary_dim = 38
        secondary_dim = bad_secondary_dim
        center_coord = np.array([63, 63])
        return_complex_moments = False
        return_pupil_mask = False
        return_reconstructed_image = False

        with pytest.raises(ZernikeValueError, match=err_str):
            zernike_features(
                image=sample_image,
                feature_type=feature_type,
                pupil_type=pupil_type,
                degree=degree,
                primary_dim=primary_dim,
                secondary_dim=secondary_dim,
                center_coord=center_coord,
                return_complex_moments=return_complex_moments,
                return_pupil_mask=return_pupil_mask,
                return_reconstructed_image=return_reconstructed_image,
            )
