import warnings
import numpy as np
from multiprocessing import cpu_count
from typing import NamedTuple

from _skimage2.measure import regionprops, label

from .zernike_cy import ZernikeFeatures, PseudoZernikeFeatures


class ZernikeResults(NamedTuple):
    """Results of Zernike feature computations as 4 attributes.

    Parameters
    ----------
    features : ndarray of shape (f,)
        Normalized Zernike features for the given input image.
    complex_moments : ndarray of shape (f,)
        Normalized complex-valued Zernike moments for the given input image.
    radius : float
        Radius value used for the circular pupil/bbox.
    center_coord : ndarray of shape (2,)
        Center value for the circular pupil/bbox.
    """

    features: np.typing.NDArray
    complex_moments: np.typing.NDArray | None = None
    radius: float = 0.0
    center_coord: np.typing.NDArray = np.array([0.0, 0.0])


def _scale_intensity(image):
    """Scale input image to [0.0, 1.0] range."""
    image = image / np.max(image)
    return image


def _compute_radius_center(image):
    """Compute Feret diameter and centroid of the labelled regionprop.

    The function is used when radius="auto" and center="auto". It assumes only one
    object is present in the given image. Assigns a label to every pixel, then uses
    ``skimage.measure.regionprops`` to compute centroid and max. Feret diameter.
    """
    image_properties = regionprops(label(image))[0]
    radius = round(image_properties.feret_diameter_max / 2.0)
    center = np.array(image_properties.centroid).round()
    return radius, center


def _convert_datatypes(image, degree, radius, center):
    """Convert all parameters to cython expected datatypes."""
    image = _scale_intensity(image)
    image = image.astype(np.float64)
    degree = int(degree)
    radius = float(radius)
    center = center.astype(np.float64)
    return image, degree, radius, center


def _verify_image(image: np.typing.NDArray):
    """Check for only binary or grayscale input image types."""
    if not isinstance(image, np.ndarray):
        raise TypeError("'image' must be a numpy array.")
    if isinstance(image, np.ndarray) and (image.ndim != 2):
        raise ValueError("Currently, only single channel images are supported.")


def _verify_feature_type(feature_type: str):
    """Check for conventional or pseudo Zernike feature choices only."""
    if not isinstance(feature_type, str):
        raise TypeError("'feature_type' can only be string.")
    if isinstance(feature_type, str) and (
        feature_type not in ("default", "dense", "pseudo")
    ):
        raise ValueError(
            "'feature_type' is not a valid choice. Choose 'default', or 'dense'/'pseudo' for denser pseudo-Zernike features."
        )


def _verify_degree(degree: int, radius: float):
    """Check for input polynomial degree value."""
    if not isinstance(degree, int):
        raise TypeError("'degree' can only be interger value.")
    if isinstance(degree, int) and (degree <= 0):
        raise ValueError("'degree' is not a valid positive integer value.")
    if isinstance(degree, int) and (degree >= radius):
        warnings.warn(
            "'degree' is a large value compared to 'radius'. Feature computations might be unstable or slow.",
            RuntimeWarning,
        )


def _verify_radius_center(
    radius: float | str, center: np.typing.NDArray | str, height, width
):
    """Check for radius and center datatype and limits."""
    radius_str_flag = False
    radius_auto_flag = False
    center_str_flag = False
    center_auto_flag = False
    if isinstance(radius, str):
        radius_str_flag = True
        if radius != "auto":
            raise ValueError("'radius' only supports 'auto' as string choice.")
        else:
            radius_auto_flag = True
    elif isinstance(radius, (int, float)):
        if (radius <= 1.0) or (radius > min(height, width)):
            raise ValueError("'radius' value is invalid for image size.")
    else:
        raise TypeError("'radius' is not a correct choice or datatype.")

    if isinstance(center, str):
        center_str_flag = True
        if center != "auto":
            raise ValueError("'center_coord' only supports 'auto' as string choice.")
        else:
            center_auto_flag = True
    elif isinstance(center, np.ndarray):
        if center.shape != (2,):
            raise ValueError("'center_coord' expects 1D array of shape '(2,)'.")
        if not radius_str_flag:
            toplim = (center[0] - radius) < 0.0
            bottomlim = (center[0] + radius) > height
            leftlim = (center[1] - radius) < 0.0
            rightlim = (center[1] + radius) > width
            if leftlim or rightlim or toplim or bottomlim:
                raise ValueError("'center_coord' and 'radius' exceed image size.")
    else:
        raise TypeError("'center_coord' is not a valid choice.")

    if (radius_str_flag or center_str_flag) and not (
        radius_str_flag and center_str_flag and radius_auto_flag and center_auto_flag
    ):
        raise ValueError(
            "Both 'radius' and 'center_coord' need to be 'auto' for automated calculations."
        )


def zernike_features(
    image: np.typing.NDArray,
    *,
    feature_type: str = "default",
    degree: int = 9,
    radius: float | str = "auto",
    center_coord: np.typing.NDArray | str = "auto",
    return_complex_moments: bool = False,
) -> ZernikeResults:
    """Extract Zernike features from a given binary or grayscale image.

    The function accepts a binary or grayscale image as input and computes
    conventional Zernike features or pseudo-Zernike features for the image.
    When using ``radius``, ``center_coord`` as "auto", then use binary image as
    input because internally it uses ``regionprops`` to compute radius, centroid.

    Parameters
    ----------
    image : ndarray of shape (M, N)
        A binary or grayscale image of size ``(M, N)``. Internally scaled to ``[0., 1.]``
        scale for feature calculation and normalization.
    feature_type : str, optional
        Choice of Zernike feature type to compute, by default ``"default"``. For
        conventional Zernike features use ``"default"``. For pseudo-Zernike features
        use either ``"pseudo"`` or ``"dense"``.
    degree : int, optional
        Zernike polynomial design parameter, by default ``9``. The highest degree of
        polynomial to compute. ``9`` gives decent starting results, but higher values
        can capture finer details of the object.
    radius : float | str, optional
        Circular pupil design parameter, by default ``"auto"``. Radius of the circular
        pupil/bbox over which features are computed.
        When using float value, the radius cannot be larger than ``min(M, N)``, and image can
        be binary or grayscale.
        When using ``"auto"``, internally, maximum Feret diameter is computed for the
        object in the whole image ``(M, N)``, hence image must be binary with one object.
        For multiple objects, use loops to provide roughly cropped images, and radius
        will be computed for each object.
    center_coord : ndarray of shape (2,) | str, optional
        Circular pupil design parameter, by default ``"auto"``. Center of the circular
        pupil/bbox in 2D image coordinates.
        When providing an ndarray, it must be of shape ``(2,)``.
        When using ``"auto"``, internally, centroid of the object in the whole image ``(M, N)``
        is calculated, hence image must be binary with one object.
    return_complex_moments : bool, optional
        Return normalized complex-valued Zernike moments, by default ``False``. Use ``True``
        to get moments, useful in image reconstruction.

    Returns
    -------
    ZernikeResults : named tuple with 4 attributes
        ``features`` : ndarray of shape (f,)
            Normalized Zernike features for the given input image.
        ``complex_moments`` : ndarray of shape (f,)
            Normalized complex-valued Zernike moments for the given input image.
        ``radius`` : float
            Radius value used for the circular pupil/bbox.
        ``center_coord`` : ndarray of shape (2,)
            Center value for the circular pupil/bbox.

    Raises
    ------
    TypeError, ValueError
        - If ``image`` is not a single-channel image of shape (M, N).
        - If ``feature_type`` is not one of ``("default", "dense", "pseudo")``.
        - If ``degree`` is not a positive integer.
        - If ``radius`` and ``center_coord`` are wrong choices or do not lie within the image.

    See Also
    --------
    skimage.measure.regionprops

    Notes
    -----
    Zernike features are shape, texture descriptors of an object within a circular bbox.
    Given an image, a unit circle centered over the image/object (circular bbox, pupil),
    Zernike features are image weighted average of Zernike polynomials. Here, Zernike
    polynomials are complex-valued trigonometric orthogonal basis functions, Zernike
    moments are normalized object pixel intensities weighted average of basis polynomials,
    and Zernike features (ZFs) are normalized magnitude of complex moments.

    As Zernike polynomials are orthogonal, ZFs can describe an image/object without
    redundancy. ZFs are rotation invariant, but as size of a circular bbox over the object
    can vary, ZFs are not scale and translation invariant. Furthermore, iterating over
    many basis polynomials can capture fine, high-frequency details of the object.
    Thus, ZFs can describe shape, structure, texture of the object.

    The design parameters for extracting ZFs for an image or object are: the highest ``degree``
    of the basis polynomial to use, and the ``radius``, ``center_coord`` of the circular bbox.
    As a rule of thumb, do not use ``degree`` values equal or larger than the ``radius`` of
    the circular pupil/bbox.

    The current implementation supports two types of Zernike features:
    1. Conventional or regular Zernike features are computed for even integers only [1]_, [4]_.
    2. Pseudo-Zernike features are computed for all integers, denser and robust to noise than ZFs [1]_, [5]_.

    If using RGB images, one option to compute ZFs is to convert RGB to grayscale
    image. Another option is to iterate over channels and compute ZFs for each channel,
    then stack the ZFs or compute mean of each feature over channels.

    References
    ----------
    .. [1] Kuo Niu and Chao Tian 2022 J. Opt. 24 123001.
            https://doi.org/10.1088/2040-8986/ac9e08
    .. [2] D. Otkupman, S. Bezdidko, V. Ostashenkova E3S Web of Conferences 310, 01002 (2021).
            https://doi.org/10.1051/e3sconf/202131001002
    .. [3] Michael Reed Teague J. Opt. Soc. Am., Vol. 70, No. 8, August 1980.
            https://doi.org/10.1364/JOSA.70.000920
    .. [4] Zernike Polynomials Wikipedia
            https://en.wikipedia.org/wiki/Zernike_polynomials
    .. [5] Pseudo Zernike Polynomials Wikipedia
            https://en.wikipedia.org/wiki/Pseudo-Zernike_polynomials
    .. [6] Image reconstruction example using Zernike moments
            https://stackoverflow.com/a/33339289

    Examples
    --------
    Below are 2 basic examples of the API usage. First is for user provided image,
    degree, radius, center. Second is for automated radius and center computation,
    but uses pseudo-Zernike as well as returns complex moments. Lastly, accessing
    ``radius`` attribute from the ``ZernikeResults`` named tuple.

    >>> import numpy as np
    >>> from skimage.feature import zernike_features
    >>> img = np.zeros((127, 127)) # create 127x127 blank image
    >>> img[38:89, 38:89] = 255 # create 50x50 white square
    >>> degree = 5
    >>> radius = 76/2 # radius slightly larger than square's diagonal
    >>> center = np.array([63, 63])
    >>> znres = zernike_features(image=img, degree=degree, radius=radius, center_coord=center)
    >>> print(znres)
    ZernikeResults(features=array([5.76335032e-01, 1.55789700e-17, 6.91282739e-01,
        3.30196655e-17, 4.34968136e-17, 3.26815381e-17, 1.27767584e-01, 4.75458231e-16,
        1.55826037e-01, 4.13079265e-17, 3.33720848e-17, 2.09353610e-17]),
        complex_moments=None, radius=38.0, center_coord=array([63., 63.]))
    >>> znres = zernike_features(image=img, feature_type="pseudo", degree=degree,
    radius="auto", center_coord="auto", return_complex_moments=True)
    >>> print(znres)
    ZernikeResults(features=array([3.47782629e-01, 1.03675405e-01, 1.86882747e-17, 3.06387237e-01,
        2.27686095e-17, 1.26912214e-16, 6.42298357e-02, 6.51737232e-18,
        1.60217837e-17, 2.23480834e-17, 9.78544850e-02, 2.08472082e-17,
        1.46213712e-16, 2.90064471e-17, 1.81683376e-01, 9.62118378e-02,
        1.09049136e-17, 3.76528544e-17, 3.17981638e-17, 1.44028202e-01,
        6.77181551e-17]), complex_moments=array([ 3.47782629e-01+0.00000000e+00j, -1.03675405e-01+0.00000000e+00j,
        -4.93067220e-19-1.86817691e-17j, -3.06387237e-01+0.00000000e+00j,
        8.29996488e-18-2.12018905e-17j, -1.26803190e-16+5.25938368e-18j,
        -6.42298357e-02+0.00000000e+00j, -2.95840332e-18+5.80723615e-18j,
        1.48452124e-17+6.02637714e-18j,  9.64220342e-18-2.01609708e-17j,
        9.78544850e-02+0.00000000e+00j,  1.09570493e-18+2.08183938e-17j,
        1.46212108e-16+6.84815584e-19j,  1.83530576e-17+2.24619512e-17j,
        -1.81683376e-01+9.79149150e-17j,  9.62118378e-02+0.00000000e+00j,
        -2.87622545e-18-1.05187674e-17j, -3.65919129e-17-8.87520997e-18j,
        1.10118346e-17+2.98305668e-17j,  1.44028202e-01-8.78218337e-17j,
        -4.76631646e-18-6.75502092e-17j]), radius=36.0, center_coord=array([63., 63.]))
    >>> print(znres.radius)
    36.0
    """
    _verify_image(image)
    _verify_feature_type(feature_type)
    height, width = image.shape[0], image.shape[1]
    _verify_radius_center(radius, center_coord, height, width)

    if (radius == "auto") and (center_coord == "auto"):
        radius, center_coord = _compute_radius_center(image)

    _verify_degree(degree, radius)

    # explicitly convert datatypes for Cython compliance
    image, degree, radius, center_coord = _convert_datatypes(
        image, degree, radius, center_coord
    )

    if feature_type in ("dense", "pseudo"):
        pznft = PseudoZernikeFeatures(
            image=image,
            degree=degree,
            radius=radius,
            center_coord=center_coord,
            num_threads=cpu_count(),
        )
        compute_features = pznft.compute_pseudo_zernike_features
    else:  # default
        znft = ZernikeFeatures(
            image=image,
            degree=degree,
            radius=radius,
            center_coord=center_coord,
            num_threads=cpu_count(),
        )
        compute_features = znft.compute_zernike_features

    normalized_features, complex_moments = compute_features()

    if not return_complex_moments:
        complex_moments = None
    znres = ZernikeResults(
        features=normalized_features,
        complex_moments=complex_moments,
        radius=radius,
        center_coord=center_coord,
    )
    return znres
