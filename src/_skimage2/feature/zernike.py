import warnings
import numpy as np
from multiprocessing import cpu_count
from typing import NamedTuple

from _skimage2.measure import regionprops, label

from .zernike_cy import ZernikeFeatures, ZernikeTypeError, ZernikeValueError

# ZernikeTypeError, ZernikeValueError are created to allow users to catch these
# errors if need be, over generic TypeError, ValueError.

# map strings to integers to pass to cython
FEATURE_OPTIONS = {"conventional": 0, "pseudo": 1}
pupil_strings = ["circle", "annulus", "ellipse", "rectangle", "square", "hexagon"]
PUPIL_OPTIONS = {v: i for i, v in enumerate(pupil_strings)}
# annulus requires (0, 1), rect, ell, annul require second dim
VALID_COMBS = {
    "circle": {"auto": {"sec_dim": None}, "numeric": {"sec_dim": None}},
    "annulus": {"auto": {"sec_dim": "range_01"}, "numeric": {"sec_dim": "lt_pd"}},
    "ellipse": {"auto": {"sec_dim": None}, "numeric": {"sec_dim": "numeric"}},
    "rectangle": {"auto": {"sec_dim": None}, "numeric": {"sec_dim": "numeric"}},
    "square": {"auto": {"sec_dim": None}, "numeric": {"sec_dim": None}},
    "hexagon": {"auto": {"sec_dim": None}, "numeric": {"sec_dim": None}},
}


class ZernikeResults(NamedTuple):
    """Results of Zernike feature computations as 6 attributes.

    Parameters
    ----------
    features : ndarray of shape (f,)
        Normalized Zernike features for the given input image.
    complex_moments : ndarray of shape (f,)
        Normalized complex-valued Zernike moments for the given input image.
    pupil_mask : ndarray of shape (M, N)
        Binary mask generated for the given pupil.
    reconstructed_image : ndarray of shape (M, N)
        Grayscale image reconstructed from the computed complex moments.
    primary_dim : float
        Primary dimension used for the given pupil shape.
    secondary_dim : float
        Secondary dimension (if any) used for the given pupil shape.
    center_coord : ndarray of shape (2,)
        Center value for the pupil.
    """

    features: np.typing.NDArray
    complex_moments: np.typing.NDArray | None = None
    pupil_mask: np.typing.NDArray | None = None
    reconstructed_image: np.typing.NDArray | None = None
    primary_dim: float = 0.0
    secondary_dim: float = 0.0
    center_coord: np.typing.NDArray = np.array([0.0, 0.0])


def _scale_intensity(image):
    """Scale input image to [0.0, 1.0] range."""
    image = image / np.max(image)
    # image = image / 255.0
    return image


def _compute_pd_sd_cc(image, pupil_type, secondary_dim):
    """Compute Feret diameter, bbox and centroid of the labelled regionprop.

    The function is used when primary_dim="auto" and center_coord="auto".
    It assumes only one object is present in the given image. Assigns a label
    to every pixel, then uses ``skimage.measure.regionprops`` to compute
    centroid, bbox, and max. Feret diameter.
    """
    # extract region from bin image
    image_properties = regionprops(label(image))[0]
    cc = np.array(image_properties.centroid).round()
    if pupil_type in ("rectangle", "square"):
        bbox_crds = image_properties.bbox
        pd = abs(bbox_crds[3] - bbox_crds[1])  # maxcol - mincol
        sd = abs(bbox_crds[2] - bbox_crds[0])  # maxrow - minrow
    elif pupil_type == "hexagon":
        # consider Feret diameter = 2*inradius, thus object will be inside hexagon
        # pd is circumradius of regular hexagon
        pd = 2.0 * (image_properties.feret_diameter_max) / np.sqrt(3)
        sd = 0.0
    elif pupil_type == "ellipse":
        if (-np.pi / 4.0) < image_properties.orientation < (np.pi / 4.0):
            # vertical ellipse
            pd = image_properties.axis_minor_length
            sd = image_properties.axis_major_length
        else:
            # horizontal ellipse
            pd = image_properties.axis_major_length
            sd = image_properties.axis_minor_length
    else:
        pd = image_properties.feret_diameter_max
        sd = secondary_dim * pd
    pd = round(pd / 2.0)
    sd = round(sd / 2.0)
    return pd, sd, cc


def _convert_datatypes(img, pt, deg, pd, sd, cc):
    """Convert all parameters to cython expected datatypes."""
    img = _scale_intensity(img)
    img = img.astype(np.float64)
    deg = int(deg)
    if pt == "square":
        # for square sec_dim is None, set pd and sd to same value
        v = max(pd, sd)
        pd = v
        sd = v
    pd = float(pd)
    sd = float(sd)
    cc = cc.astype(np.float64)
    return img, deg, pd, sd, cc


def _check_auto(x):
    """Helper for checking 'auto' inputs."""
    return isinstance(x, str) and (x == "auto")


def _check_num_range(x, l, h):
    """Helper for checking number inputs are in range."""
    return isinstance(x, (int, float)) and (l < x < h)


def _check_arr_2(x):
    """Helper for checking 'center_coord' if ndarray of (2,)."""
    return isinstance(x, np.ndarray) and (x.shape == (2,))


def _verify_image(img):
    """Check for only single channel binary or grayscale input image types."""
    arr_flag = isinstance(img, np.ndarray)
    if not arr_flag:
        raise ZernikeTypeError("'image' must be a numpy array.")
    if arr_flag and (img.ndim != 2):
        raise ZernikeValueError("Only single channel images are supported.")


def _verify_feature_type(ft):
    """Check for conventional or pseudo Zernike feature choices."""
    str_flag = isinstance(ft, str)
    if not str_flag:
        raise ZernikeTypeError("'feature_type' can only be string.")
    if str_flag and (ft not in FEATURE_OPTIONS):
        raise ZernikeValueError(
            "'feature_type' is not a valid choice. Choose 'conventional' or 'pseudo'."
        )


def _verify_degree(deg, pd, sd):
    """Check for input polynomial degree value."""
    deg_int_flag = isinstance(deg, int)
    # when deg greater than size of pupil, Zernike overfits noise.
    # warn user when degree is higher than one of the smaller dimension
    limit = min(pd, sd) if sd > 1.0 else pd
    if not deg_int_flag:
        raise ZernikeTypeError("'degree' can only be interger value.")
    if deg_int_flag and (deg <= 0):
        raise ZernikeValueError("'degree' is not a valid positive integer value.")
    if deg_int_flag and (deg >= limit):
        warnings.warn(
            "'degree' value is large compared to pupil dimensions. Feature computations might be unstable and slow.",
            RuntimeWarning,
        )


def _verify_pt_pd_sd_cc(pt, pd, sd, cc, height, width):
    """Check for primary, secondary and center datatype and limits."""
    pt_str_flag = isinstance(pt, str)
    pd_type = None
    pdll = 1
    sdll = 1

    if not pt_str_flag:
        raise ZernikeTypeError("'pupil_type' is not a string.")
    if pt_str_flag and (pt not in VALID_COMBS):
        raise ZernikeValueError(
            "'pupil_type' is not a valid pupil. Choose one of 6 pupil shapes."
        )

    if pt == "rectangle":
        # for rectangle, pupil can extend upto whole image
        pdul = max(height, width)
    else:
        # otherwise limit to smaller of dimension
        pdul = min(height, width)
    sdul = pdul

    if _check_auto(pd):
        pd_type = "auto"
        if not _check_auto(cc):
            raise ZernikeValueError(
                "When 'primary_dim='auto'', 'center_coord' must be 'auto'."
            )
    elif _check_num_range(pd, pdll, pdul):
        # check pd lies within image
        pd_type = "numeric"
        if not _check_arr_2(cc):
            raise ZernikeValueError("'center_coord' must be 1D array of shape (2,).")
        else:
            toplim = (cc[0] - pd) < 0.0
            bottomlim = (cc[0] + pd) > height
            leftlim = (cc[1] - pd) < 0.0
            rightlim = (cc[1] + pd) > width
            if leftlim or rightlim or toplim or bottomlim:
                raise ZernikeValueError(
                    "'center_coord' and 'primary_dim' exceed image size. Shift the center or reduce primary_dim."
                )
    else:
        raise ZernikeTypeError("'primary_dim' can only be 'auto' or a number.")

    exp_sec_dim = VALID_COMBS[pt][pd_type]["sec_dim"]

    if exp_sec_dim is None:
        if sd is not None:
            raise ZernikeValueError("'secondary_dim' is expected to be set to 'None'.")
    elif exp_sec_dim == "range_01":
        # checking for annulus when pd auto
        if not _check_num_range(sd, 0.0, 1.0):
            raise ZernikeValueError(
                "For selected pupil, 'secondary_dim' must be in range interval (0.0, 1.0)."
            )
    elif exp_sec_dim == "lt_pd":
        if not _check_num_range(sd, 1.0, pd):
            # expect at least 1 pixel obscure radius
            raise ZernikeValueError(
                "For selected pupil, 'secondary_dim' must be in range interval (1.0, 'primary_dim')."
            )
    elif exp_sec_dim == "numeric":
        # check sd lies within image
        if not _check_num_range(sd, sdll, sdul):
            raise ZernikeValueError(
                "'secondary_dim' exceed image size. Reduce secondary_dim."
            )
    else:
        # added to avoid dangling else.
        # most likely this will never be raised as VALID_COMBS always gets correct keys.
        raise ZernikeTypeError(
            "'secondary_dim' can only be 'None', between (0, 1), or number."
        )


def _verify_return_flags(rcm, rpm, rci):
    """Check if return flags are boolean values."""
    if not isinstance(rcm, bool):
        raise ZernikeTypeError("'return_complex_moments' can only be boolean.")
    if not isinstance(rpm, bool):
        raise ZernikeTypeError("'return_pupil_mask' can only be boolean.")
    if not isinstance(rci, bool):
        raise ZernikeTypeError("'return_reconstructed_image' can only be boolean.")


def _validate_parse_args(
    image,
    feature_type,
    pupil_type,
    degree,
    primary_dim,
    secondary_dim,
    center_coord,
    return_complex_moments,
    return_pupil_mask,
    return_reconstructed_image,
):
    """Validate all the arguments for interface API and convert to appropriate datatypes."""
    _verify_image(image)
    _verify_feature_type(feature_type)
    _verify_return_flags(
        return_complex_moments, return_pupil_mask, return_reconstructed_image
    )
    height, width = image.shape[0], image.shape[1]
    _verify_pt_pd_sd_cc(
        pupil_type, primary_dim, secondary_dim, center_coord, height, width
    )
    # convert to number to avoid nonetype errors
    secondary_dim = 0.0 if secondary_dim is None else secondary_dim
    if (primary_dim == "auto") and (center_coord == "auto"):
        primary_dim, secondary_dim, center_coord = _compute_pd_sd_cc(
            image, pupil_type, secondary_dim
        )
    _verify_degree(degree, primary_dim, secondary_dim)
    # explicitly convert datatypes compliant with Cython
    image, degree, primary_dim, secondary_dim, center_coord = _convert_datatypes(
        image, pupil_type, degree, primary_dim, secondary_dim, center_coord
    )
    return image, degree, primary_dim, secondary_dim, center_coord


def zernike_features(
    image,
    *,
    feature_type="conventional",
    degree=9,
    pupil_type="circle",
    primary_dim="auto",
    secondary_dim=None,
    center_coord="auto",
    return_complex_moments=False,
    return_pupil_mask=False,
    return_reconstructed_image=False,
):
    """Extract Zernike features from a given binary or grayscale image.

    Extract conventional Zernike features or pseudo-Zernike features from a
    binary or grayscale image. When using ``primary_dim``, ``center_coord``
    as "auto", internally it uses ``regionprops`` to compute
    max. Feret radius, bbox, centroid.

    Valid input parameter combinations are as follows:

    +-----------------+-----------------+-------------------------+--------------------+---------------------+
    | ``pupil_type``  | ``primary_dim`` | ``secondary_dim``       | ``center_coord``   | ``image``           |
    +=================+=================+=========================+====================+=====================+
    | ``"circle"``    | ``"auto"``      | ``None``                | ``"auto"``         | Binary image        |
    +-----------------+-----------------+-------------------------+--------------------+---------------------+
    | ``"circle"``    | ``float``       | ``None``                | 1D array ``(2,)``  | Binary or grayscale |
    +-----------------+-----------------+-------------------------+--------------------+---------------------+
    | ``"annulus"``   | ``"auto"``      | ``float`` in ``(0, 1)`` | ``"auto"``         | Binary image        |
    +-----------------+-----------------+-------------------------+--------------------+---------------------+
    | ``"annulus"``   | ``float``       | ``float``               | 1D array ``(2,)``  | Binary or grayscale |
    +-----------------+-----------------+-------------------------+--------------------+---------------------+
    | ``"ellipse"``   | ``"auto"``      | ``None``                | ``"auto"``         | Binary image        |
    +-----------------+-----------------+-------------------------+--------------------+---------------------+
    | ``"ellipse"``   | ``float``       | ``float``               | 1D array ``(2,)``  | Binary or grayscale |
    +-----------------+-----------------+-------------------------+--------------------+---------------------+
    | ``"rectangle"`` | ``"auto"``      | ``None``                | ``"auto"``         | Binary image        |
    +-----------------+-----------------+-------------------------+--------------------+---------------------+
    | ``"rectangle"`` | ``float``       | ``float``               | 1D array ``(2,)``  | Binary or grayscale |
    +-----------------+-----------------+-------------------------+--------------------+---------------------+
    | ``"square"``    | ``"auto"``      | ``None``                | ``"auto"``         | Binary image        |
    +-----------------+-----------------+-------------------------+--------------------+---------------------+
    | ``"square"``    | ``float``       | ``None``                | 1D array ``(2,)``  | Binary or grayscale |
    +-----------------+-----------------+-------------------------+--------------------+---------------------+
    | ``"hexagon"``   | ``"auto"``      | ``None``                | ``"auto"``         | Binary image        |
    +-----------------+-----------------+-------------------------+--------------------+---------------------+
    | ``"hexagon"``   | ``float``       | ``None``                | 1D array ``(2,)``  | Binary or grayscale |
    +-----------------+-----------------+-------------------------+--------------------+---------------------+

    Parameters
    ----------
    image : ndarray of shape (M, N)
        A binary or grayscale image of size ``(M, N)``. Internally scaled to ``[0., 1.]``
        scale for feature calculation and normalization.
    feature_type : str, optional
        Choice of Zernike feature type to compute, by default ``"conventional"``.
        For conventional Zernike features use ``"conventional"``.
        For pseudo-Zernike features use ``"pseudo"``.
    degree : int, optional
        Zernike polynomial degree parameter, by default ``9``. The highest degree of
        polynomial to compute. Higher values can capture finer details of the object,
        but may take longer. Raises ``RuntimeWarning`` when degree is greater than pupil size.
    pupil_type : str, optional
        Choice of pupil shape to use for Zernike orthonormal basis, by default ``"circle"``.
        Circular pupil uses Zernike polynomials which are orthonormal over circle.
        Non-circular pupils use Gram-matrix, Cholesky decomposition to generate
        Zernike-equivalent orthonormal basis over the pupil grid.
        Use ``"circle"`` for circular pupil. ``"annulus"`` for annular/ring-like pupil.
        ``"ellipse"`` for elliptical pupil where major-axis can only be horizontal or vertical.
        ``"rectangle"`` for rectangular pupil. ``"square"`` for square pupil. ``"hexagon"``
        for regular hexagonal pupil with top and bottom flat orientation.
    primary_dim : float | str, optional
        Pupil's primary dimension, by default ``"auto"``.
        For circular, annular pupil it is radius. For ellipse it is semi-major axis.
        For rectangle it is half-width. For square it is half-side. For regular hexagon
        it is side, equivalent to inscrbing inradius. When using ``"auto"``, internally,
        max. Feret diameter or bbox is computed for the object in the whole image ``(M, N)``,
        hence image must be binary with one object.
    secondary_dim : float | str, optional
        Pupil's secondary dimension, by default ``None``.
        For circular pupil it is not used. For annular pupil it is obscure radius.
        For ellipse it is semi-minor axis. For rectangle it is semi-height.
        For square it is copied from ``primary_dim``. For regular hexagon it is not used.
        When ``primary_dim="auto"``, ``secondary_dim`` is obscuration ratio for
        annular pupil. Hence ``secondary_dim`` must be provided for annulus as a
        value in open interval ``(0.0, 1.0)``.
    center_coord : ndarray of shape (2,) | str, optional
        Center of the pupil, by default ``"auto"``. Center of the pupil in 2D image
        coordinates. When providing an ndarray, it must be of shape ``(2,)``.
        When using ``"auto"``, internally, centroid of the object in the whole image
        ``(M, N)`` is calculated, hence image must be binary with one object.
    return_complex_moments : bool, optional
        Return normalized complex-valued Zernike moments, by default ``False``.
    return_pupil_mask : bool, optional
        Return pupil mask generated and used for image Zernike moments, by default ``False``.
    return_reconstructed_image : bool, optional
        Return grayscale reconstructed image, by default ``False``. Uses orthonormal
        basis and complex moments to reconstruct the image i.e. to compute
        complex-valued pixel intensities inside the pupil. Returns the real part
        of the complex pixels as grayscale image.

    Returns
    -------
    ZernikeResults : named tuple with 6 attributes
        ``features`` : ndarray of shape (f,)
            Normalized Zernike features for the given input image.
        ``complex_moments`` : ndarray of shape (f,)
            Normalized complex-valued Zernike moments for the given input image.
        ``pupil_mask`` : ndarray of shape (M, N)
            Binary mask generated for the given pupil.
        ``reconstructed_image`` : ndarray of shape (M, N)
            Grayscale image reconstructed from the computed complex moments.
        ``primary_dim`` : float
            Primary dimension used for the given pupil shape.
        ``secondary_dim`` : float
            Secondary dimension if used for the given pupil shape, else 0.0.
        ``center_coord`` : ndarray of shape (2,)
            Center value for the pupil.

    Raises
    ------
    ZernikeTypeError, ZernikeValueError
        - If ``image`` is not a single-channel image of shape (M, N).
        - If ``feature_type`` is not one of ``("conventional", "pseudo")``.
        - If ``pupil_type`` is not one of 6 pupil shape.
        - If ``degree`` is not a positive integer.
        - If ``primary_dim``, ``secondary_dim`` and ``center_coord`` are wrong choices or do not lie within the image.
        - If Cholesky decomposition and inverse computation fails for non-circular pupils.

    RuntimeWarning
        - If ``degree`` greater than pupil size` as it fits very high-frequency polynomials over pixel-to-pixel variations that captures noise.

    See Also
    --------
    skimage.measure.regionprops

    Notes
    -----
    Zernike features are shape, texture descriptors of an object within a pupil.
    Given an image, a unit circle centered over the image/object (circular bbox, pupil),
    Zernike features are image weighted average of Zernike polynomials. Here, Zernike
    polynomials are complex-valued trigonometric orthogonal basis functions, Zernike
    moments are normalized pixel intensities weighted average of basis polynomials,
    and Zernike features (ZFs) are normalized magnitude of complex moments. [5]_

    As Zernike polynomials are orthogonal, ZFs can describe an image/object without
    redundancy. ZFs are rotation invariant, but as size of a circular pupil over the object
    can vary, ZFs are not scale and translation invariant. Furthermore, iterating over
    many basis polynomials can capture fine, high-frequency details of the object.
    Thus, ZFs can describe shape, structure, texture of the object. [1]_ [5]_

    The current implementation supports two types of Zernike features:

    1. Conventional or regular Zernike features are computed for even integers only. [1]_ [2]_ [4]_
    2. Pseudo-Zernike features are computed for all integers, denser and robust to noise than ZFs. [1]_ [5]_

    The design parameters for extracting ZFs for an image or object are: the highest ``degree``
    of the basis polynomial, the pupil size ``primary_dim`` and ``secondary_dim``,
    and center of the pupil ``center_coord``.
    As a rule of thumb, do not use ``degree`` greater than the size of the pupil as
    the polynomials will start overfitting to noise and provide bad features.

    There are 6 pupil shapes supported: circle, annulus, ellipse, rectangle, square, hexagon.
    To use these, refer to the table in API docs.

    Zernike polynomials, as per their definition, are orthogonal only over unit-circle pupils.
    For arbitrary non-circular pupils, these Zernike polynomials need to be orthonormalized over
    the pupil domain. This process is performed numerically using non-recursive matrix method.
    It starts by evaluating Zernike polynomials over the pupil coordinates (x, y). It then
    computes Gram-matrix (necessary for normalization) and Cholesky decomposition of Gram-matrix
    (necessay for orthogonalization), then transforms the evaluated Zernike polynomials
    with inverse of Cholesky decomposed matrix to compute Zernike-like orthonormal basis.
    The product of these orthonormal basis with image pixel intensities are the orthonormalized,
    complex-valued Zernike-like moments. The features are then magnitude of these complex-moments. [1]_ [2]_ [6]_

    In the limiting case of annular pupil with ``secondary_dim=0.0``, or elliptical pupil with
    ``secondary_dim=primary_dim`` where the pupil becomes circular, the complex-moments
    and features, computed using either orthogonal Zernike polynomials or matrix method,
    are equaivalent. The slight differences are due to floating-point values and vanishingly
    small off-diagonal non-zero values in the Gram-matrix computation. [3]_ [2]_

    If using RGB images, one option to compute ZFs is to convert RGB to grayscale
    image. Another option is to iterate over channels and compute ZFs for each channel,
    then stack the ZFs and use PCA or such dimensionality reduction technique.

    References
    ----------
    .. [1] Kuo Niu and Chao Tian 2022 J. Opt. 24 123001. https://doi.org/10.1088/2040-8986/ac9e08
    .. [2] Guang-ming Dai and Virendra N. Mahajan, Opt. Lett. 32, 74-76 (2007).https://doi.org/10.1364/OL.32.000074
    .. [3] Churui Li, Gongjian Guo, and Xiang Hao, J. Opt. Soc. Am. A 43, 1053-1066 (2026). https://doi.org/10.1364/JOSAA.601210
    .. [4] Dmitriy Otkupman, Sergey Bezdidko and Victoria Ostashenkova E3S Web Conf., 310 (2021) 01002. https://doi.org/10.1051/e3sconf/202131001002D
    .. [5] Michael Reed Teague, J. Opt. Soc. Am. 70, 920-930 (1980). https://doi.org/10.1364/JOSA.70.000920
    .. [6] Cosmas Mafusire and Tjaart P. J. Krüger, J. Opt. Soc. Am. A 35, 840-849 (2018). https://doi.org/10.1364/JOSAA.35.000840
    .. [7] Charles E. Campbell, J. Opt. Soc. Am. A 20, 209-217 (2003). https://doi.org/10.1364/JOSAA.20.000209
    .. [8] Zernike Polynomials Wikipedia: https://en.wikipedia.org/wiki/Zernike_polynomials
    .. [9] Pseudo Zernike Polynomials Wikipedia: https://en.wikipedia.org/wiki/Pseudo-Zernike_polynomials
    .. [10] Image reconstruction example using Zernike moments: https://stackoverflow.com/a/33339289
    .. [11] Gram matrix Wikipedia: https://en.wikipedia.org/wiki/Gram_matrix
    .. [12] Cholesky decomposition Wikipedia: https://en.wikipedia.org/wiki/Cholesky_decomposition

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.feature import zernike_features
    >>> img = np.zeros((127, 127))  # create 127x127 blank image
    >>> img[38:89, 38:89] = 255  # create 50x50 white square
    >>> zf = zernike_features(img, degree=5, primary_dim=38.0, center_coord=np.array([63, 63]))
    >>> print(zf.features)
    [5.76335032e-01 1.25962373e-16 6.91282739e-01 1.62363660e-18
    1.25984971e-16 8.36823941e-18 1.27767584e-01 1.11472819e-16
    1.55826037e-01 7.57203629e-17 7.72446795e-17 3.59631500e-17]
    >>> zf = zernike_features(image=img, feature_type="pseudo", pupil_type="square", degree=5)
    >>> print(zf.primary_dim)
    26.0

    """
    image, degree, primary_dim, secondary_dim, center_coord = _validate_parse_args(
        image,
        feature_type,
        pupil_type,
        degree,
        primary_dim,
        secondary_dim,
        center_coord,
        return_complex_moments,
        return_pupil_mask,
        return_reconstructed_image,
    )

    zf = ZernikeFeatures(
        image=image,
        feature_type=FEATURE_OPTIONS[feature_type],
        pupil_type=PUPIL_OPTIONS[pupil_type],
        degree=degree,
        primary_dim=primary_dim,
        secondary_dim=secondary_dim,
        center_coord=center_coord,
        num_threads=cpu_count(),
    )
    res_dict = zf.compute_features()
    normalized_features = res_dict["fts"]
    complex_moments = res_dict["cms"]
    pupil_mask = res_dict["pmk"]
    reconstructed_image = res_dict["rim"]

    if not return_complex_moments:
        complex_moments = None
    if not return_pupil_mask:
        pupil_mask = None
    if not return_reconstructed_image:
        reconstructed_image = None

    zfres = ZernikeResults(
        features=normalized_features,
        complex_moments=complex_moments,
        pupil_mask=pupil_mask,
        reconstructed_image=reconstructed_image,
        primary_dim=primary_dim,
        secondary_dim=secondary_dim,
        center_coord=center_coord,
    )
    return zfres
