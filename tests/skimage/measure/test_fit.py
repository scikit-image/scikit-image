import numpy as np
import pytest

from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
    arch32,
    is_wasm,
    assert_almost_equal,
    assert_array_equal,
    assert_array_less,
    assert_equal,
    xfail,
    assert_stacklevel,
)
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials, add_from_estimate
from skimage.transform import AffineTransform

MODEL_CLASSES = CircleModel, EllipseModel, LineModelND


DUMMY_LINE_ARGS = ((0, 0), (1, 1))
DUMMY_CIRCLE_ARGS = ((0, 0), 1)
DUMMY_ELLIPSE_ARGS = ((0, 0), (1, 2), 0.2)


def test_line_model_predict():
    model = LineModelND(*DUMMY_LINE_ARGS)
    x = np.arange(-10, 10)
    y = model.predict_y(x)
    assert_almost_equal(x, model.predict_x(y))


def test_line_model_nd_invalid_input():
    # Test combinations of init and params.

    # A default model.
    default_lm = LineModelND(*DUMMY_LINE_ARGS)
    x = default_lm.predict_x(np.zeros(1))
    assert x == 0

    # A model with params == None.  When our init deprecations expire, this
    # will be hard to produce.
    with pytest.warns(FutureWarning, match='Calling ``LineModelND'):
        none_model = LineModelND()

    scalar = np.zeros(1)
    data2d = np.zeros((4, 2))
    for meth, inp in (
        (none_model.predict_x, scalar),
        (none_model.predict_y, scalar),
        (none_model.residuals, data2d),
    ):
        # We need extra parameters to predict for the None model.
        with testing.raises(ValueError):
            meth(inp)

        # If we do pass parameters as `params`, they need to match init.
        with testing.raises(ValueError, match='Input `params` should be length 2'):
            with pytest.warns(FutureWarning, match='Parameter `params` is deprecated'):
                meth(inp, np.zeros(1))

    tf = LineModelND.from_estimate(np.ones((1, 3)))
    assert not tf
    assert str(tf) == 'LineModelND: estimate under-determined'
    tf = LineModelND.from_estimate(np.ones((1, 2)))
    assert not tf
    assert str(tf) == 'LineModelND: estimate under-determined'

    with testing.raises(ValueError, match='`origin` is 2D, but `data` is 3D'):
        default_lm.residuals(np.zeros((1, 3)))

    # `origin` and `direction` must be of same length.
    with testing.raises(ValueError):
        LineModelND([0, 0], [1, 1, 1])
    with testing.raises(ValueError):
        LineModelND([0, 0, 0], [1, 1])


def test_line_model_nd_predict():
    origin, direction = np.array([0, 0]), np.array([0.2, 0.8])
    model = LineModelND(origin, direction)
    x = np.arange(-10, 10)
    y = model.predict_y(x)
    assert_almost_equal(x, model.predict_x(y))
    # Same result from passing params into null object.
    with pytest.warns(FutureWarning, match='Calling'):
        null_model = LineModelND()
    with pytest.warns(FutureWarning, match='Parameter'):
        assert_almost_equal(x, null_model.predict_x(y, (origin, direction)))


def test_line_model_nd_estimate():
    # generate original data without noise
    model0 = LineModelND(
        np.array([0, 0, 0], dtype='float'),
        np.array([1, 1, 1], dtype='float') / np.sqrt(3),
    )
    # we scale the unit vector with a factor 10 when generating points on the
    # line in order to compensate for the scale of the random noise
    data0 = (
        model0.origin + 10 * np.arange(-100, 100)[..., np.newaxis] * model0.direction
    )

    # add gaussian noise to data
    rng = np.random.default_rng(1234)
    data = data0 + rng.normal(size=data0.shape)

    # estimate parameters of noisy data
    model_est = LineModelND.from_estimate(data)
    # assert_almost_equal(model_est.residuals(data0), np.zeros(len(data)), 1)

    # test whether estimated parameters are correct
    # we use the following geometric property: two aligned vectors have
    # a cross-product equal to zero
    # test if direction vectors are aligned
    assert_almost_equal(
        np.linalg.norm(np.cross(model0.direction, model_est.direction)), 0, 1
    )
    # test if origins are aligned with the direction
    a = model_est.origin - model0.origin
    if np.linalg.norm(a) > 0:
        a /= np.linalg.norm(a)
    assert_almost_equal(np.linalg.norm(np.cross(model0.direction, a)), 0, 1)

    # With estimate method.
    with pytest.warns(FutureWarning, match='Calling ``LineModelND'):
        model2 = LineModelND()
    with pytest.warns(FutureWarning, match='`estimate` is deprecated') as record:
        assert model2.estimate(data)
    assert_stacklevel(record)
    assert len(record) == 1
    assert_array_equal(model2.origin, model_est.origin)
    assert_array_equal(model2.direction, model_est.direction)


def test_line_model_nd_residuals():
    model = LineModelND(np.array([0, 0, 0]), np.array([0, 0, 1]))
    assert_equal(abs(model.residuals(np.array([[0, 0, 0]]))), 0)
    assert_equal(abs(model.residuals(np.array([[0, 0, 1]]))), 0)
    assert_equal(abs(model.residuals(np.array([[10, 0, 0]]))), 10)
    # Test params argument in model.rediduals.
    data = np.array([[10, 0, 0]])
    model = LineModelND(np.array([0, 0, 0]), np.array([2, 0, 0]))
    assert_equal(abs(model.residuals(data)), 30)


def test_circle_model_invalid_input():
    # A valid default model.
    default_model = CircleModel((0, 0), 1)
    # Predict a couple of points.
    angles = [0, np.pi / 2]
    assert_almost_equal(default_model.predict_xy(angles), [[1, 0], [0, 1]])

    # A model with params == None.  When our init deprecations expire, this
    # will be hard to produce.
    with pytest.warns(FutureWarning, match='Calling ``CircleModel'):
        none_model = CircleModel()

    # We need extra parameters to predict for the None model.
    with testing.raises(ValueError):
        none_model.predict_xy(angles)

    # If we do pass parameters as `params`, they need to match init.
    for params in (
        np.zeros(1),  # center should be length 2.
        np.zeros(2),  # Need radius.
    ):
        with testing.raises(ValueError, match='Input `params` should be length 3'):
            with pytest.warns(FutureWarning, match='Parameter `params` is deprecated'):
                none_model.predict_xy(angles, params)

    with testing.raises(ValueError):
        CircleModel.from_estimate(np.empty((5, 3)))

    with testing.raises(ValueError):
        with pytest.warns(FutureWarning, match='`estimate` is deprecated') as record:
            none_model.estimate(np.empty((5, 3)))
    assert_stacklevel(record)
    assert len(record) == 1

    # Center must be 2D
    with testing.raises(TypeError):
        CircleModel(0, 1)
    with testing.raises(ValueError):
        CircleModel([0, 0, 0], 1)


def test_circle_model_predict():
    model = CircleModel((0, 0), 5)
    t = np.arange(0, 2 * np.pi, np.pi / 2)

    xy = np.array(((5, 0), (0, 5), (-5, 0), (0, -5)))
    assert_almost_equal(xy, model.predict_xy(t))


def test_circle_model_estimate():
    # generate original data without noise
    model0 = CircleModel((10, 12), 3)
    t = np.linspace(0, 2 * np.pi, 1000)
    data0 = model0.predict_xy(t)

    # add gaussian noise to data
    rng = np.random.default_rng(1234)
    data = data0 + rng.normal(size=data0.shape)

    # estimate parameters of noisy data (from_estimate method).
    model_est = CircleModel.from_estimate(data)

    # test whether estimated parameters almost equal original parameters
    assert_almost_equal(model0.center, model_est.center, 0)
    assert_almost_equal(model0.radius, model_est.radius, 0)

    # estimate method.
    with pytest.warns(FutureWarning, match='Calling ``CircleModel'):
        model_est2 = CircleModel()
    with pytest.warns(FutureWarning, match='`estimate` is deprecated') as record:
        assert model_est2.estimate(data)
    assert_stacklevel(record)
    assert len(record) == 1
    assert_array_equal(model_est2.center, model_est.center)
    assert_array_equal(model_est2.radius, model_est.radius)


def test_circle_model_int_overflow():
    xy = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int32)
    xy += 500

    model = CircleModel.from_estimate(xy)
    assert_almost_equal(model.center, [500, 500])
    assert_almost_equal(model.radius, 1)

    # estimate method.
    with pytest.warns(FutureWarning, match='Calling ``CircleModel'):
        model2 = CircleModel()
    with pytest.warns(FutureWarning, match='`estimate` is deprecated') as record:
        assert model2.estimate(xy)
    assert_stacklevel(record)
    assert len(record) == 1
    assert_almost_equal(model2.center, [500, 500])
    assert_almost_equal(model2.radius, 1)


def test_circle_model_residuals():
    model = CircleModel((0, 0), 5)
    assert_almost_equal(abs(model.residuals(np.array([[5, 0]]))), 0)
    assert_almost_equal(abs(model.residuals(np.array([[6, 6]]))), np.sqrt(2 * 6**2) - 5)
    assert_almost_equal(abs(model.residuals(np.array([[10, 0]]))), 5)


@pytest.mark.parametrize(
    "data",
    (
        [[1, 2], [3, 4]],
        [[0, 0], [1, 1], [2, 2]],
    ),
)
def test_circle_model_insufficient_data(data):
    msg = "Input does not contain enough significant data points."
    dep_msg = '`estimate` is deprecated'

    data = np.array(data)
    tf = CircleModel.from_estimate(data)
    assert not tf
    assert str(tf).endswith(msg)

    # Deprecated estimate warning.
    tf = CircleModel(*DUMMY_CIRCLE_ARGS)
    with expected_warnings([dep_msg, msg]) as _warnings:
        assert tf.estimate(data)
    assert_stacklevel(_warnings)


def test_circle_model_std_too_small():
    msg = (
        "Standard deviation of data is too small to estimate "
        "circle with meaningful precision."
    )
    dep_msg = '`estimate` is deprecated'

    data = np.ones((6, 2))
    tf = CircleModel.from_estimate(data)
    assert not tf and str(tf).endswith(msg)

    tf = CircleModel(*DUMMY_CIRCLE_ARGS)
    with pytest.warns(FutureWarning, match=dep_msg):
        with pytest.warns(RuntimeWarning, match=msg) as _warnings:
            assert tf.estimate(data)
    assert_stacklevel(_warnings)
    assert len(_warnings) == 2


def test_circle_model_estimate_from_small_scale_data():
    params = np.array([1.23e-90, 2.34e-90, 3.45e-100], dtype=np.float64)
    center, radius = params[:2], params[2]
    angles = np.array(
        [
            0.107,
            0.407,
            1.108,
            1.489,
            2.216,
            2.768,
            3.183,
            3.969,
            4.840,
            5.387,
            5.792,
            6.139,
        ],
        dtype=np.float64,
    )
    data = CircleModel(center, radius).predict_xy(angles)
    # assert that far small scale data can be estimated
    float_data = data.astype(np.float64)
    model = CircleModel.from_estimate(float_data)
    # test whether the predicted parameters are close to the original ones
    assert_almost_equal(center, model.center)
    assert_almost_equal(radius, model.radius)
    # estimate method
    model = CircleModel(*DUMMY_CIRCLE_ARGS)
    with pytest.warns(FutureWarning, match='`estimate` is deprecated') as record:
        assert model.estimate(float_data)
    assert_stacklevel(record)
    assert len(record) == 1
    assert_almost_equal(center, model.center)
    assert_almost_equal(radius, model.radius)


def _assert_ellipse_equal(model, model2):
    assert_array_equal(model.center, model2.center)
    assert_array_equal(model.axis_lengths, model2.axis_lengths)
    assert_array_equal(model.theta, model2.theta)


def _assert_ellipse_almost_equal(model, model2):
    assert_almost_equal(model.center, model2.center)
    assert_almost_equal(model.axis_lengths, model2.axis_lengths)
    assert_almost_equal(model.theta, model2.theta)


def test_ellipse_model_invalid_input():
    # A valid default model, in fact corresponding to a unit circle.
    default_model = EllipseModel((0, 0), (1, 1), 0)
    # Predict a couple of points.
    angles = [0, np.pi / 2]
    assert_almost_equal(default_model.predict_xy(angles), [[1, 0], [0, 1]])

    # A model with params == None.  When our init deprecations expire, this
    # will be hard to produce.
    with pytest.warns(FutureWarning, match='Calling ``EllipseModel'):
        none_model = EllipseModel()

    # We need extra parameters to predict for the None model.
    with testing.raises(ValueError):
        none_model.predict_xy(angles)

    # If we do pass parameters as `params`, they need to match init.
    for params in (
        np.zeros(1),  # center should be length 2.
        np.zeros(2),  # Need first axis length.
        [0, 0, 1],  # Need second axis length.
        [0, 0, 1, 1],  # Need theta
    ):
        with testing.raises(ValueError, match='Input `params` should be length 5'):
            with pytest.warns(FutureWarning, match='Parameter `params` is deprecated'):
                none_model.predict_xy(angles, params)

    with testing.raises(ValueError):
        EllipseModel.from_estimate(np.empty((5, 3)))

    with testing.raises(ValueError, match='Center coordinates should be length 2'):
        EllipseModel((0,), (1, 1), 0)
    with testing.raises(ValueError, match='Axis lengths should be length 2'):
        EllipseModel((0, 0), (1,), 0)


def test_ellipse_model_predict():
    model = EllipseModel((0, 0), (5, 10), 0)
    t = np.arange(0, 2 * np.pi, np.pi / 2)

    xy = np.array(((5, 0), (0, 10), (-5, 0), (0, -10)))
    assert_almost_equal(xy, model.predict_xy(t))


@pytest.mark.parametrize('angle', range(0, 180, 15))
def test_ellipse_model_estimate(angle):
    rad = np.deg2rad(angle)
    # generate original data without noise
    model0 = EllipseModel((10, 20), (15, 25), rad)
    t = np.linspace(0, 2 * np.pi, 100)
    data0 = model0.predict_xy(t)

    # add gaussian noise to data
    rng = np.random.default_rng(1234)
    data = data0 + rng.normal(size=data0.shape)

    # estimate parameters of noisy data
    model_est = EllipseModel.from_estimate(data)

    # Test whether estimated parameters almost equal original parameters.
    assert_almost_equal(model0.center, model_est.center, 0)
    res = model_est.residuals(data0)
    assert_array_less(res, np.ones(res.shape))

    # Estimate method.
    with pytest.warns(FutureWarning, match='Calling ``EllipseModel'):
        model_est2 = EllipseModel()
    with pytest.warns(FutureWarning, match='`estimate` is deprecated') as record:
        assert model_est2.estimate(data)
    assert_stacklevel(record)
    assert len(record) == 1
    _assert_ellipse_equal(model_est, model_est2)


@pytest.mark.parametrize('angle', np.arange(0, 180 + 1, 1))
def test_ellipse_parameter_stability(angle):
    """The fit should be modified so that a > b"""

    # generate rotation matrix
    theta = np.deg2rad(angle)
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    # generate points on ellipse
    t = np.linspace(0, 2 * np.pi, 20)
    a = 100
    b = 50
    points = np.array([a * np.cos(t), b * np.sin(t)])
    points = R @ points

    # fit model to points
    ellipse_model = EllipseModel.from_estimate(points.T)
    (a_prime, b_prime) = ellipse_model.axis_lengths
    theta_prime = ellipse_model.theta

    assert_almost_equal(theta_prime, theta)
    assert_almost_equal(a_prime, a)
    assert_almost_equal(b_prime, b)

    # Estimate method
    ellipse_model2 = EllipseModel(*DUMMY_ELLIPSE_ARGS)
    with pytest.warns(FutureWarning, match='`estimate` is deprecated') as record:
        assert ellipse_model2.estimate(points.T)
    assert_stacklevel(record)
    assert len(record) == 1
    _assert_ellipse_equal(ellipse_model, ellipse_model2)


def test_ellipse_model_estimate_from_data():
    data = np.array(
        [
            [264, 854],
            [265, 875],
            [268, 863],
            [270, 857],
            [275, 905],
            [285, 915],
            [305, 925],
            [324, 934],
            [335, 764],
            [336, 915],
            [345, 925],
            [345, 945],
            [354, 933],
            [355, 745],
            [364, 936],
            [365, 754],
            [375, 745],
            [375, 735],
            [385, 736],
            [395, 735],
            [394, 935],
            [405, 727],
            [415, 736],
            [415, 727],
            [425, 727],
            [426, 929],
            [435, 735],
            [444, 933],
            [445, 735],
            [455, 724],
            [465, 934],
            [465, 735],
            [475, 908],
            [475, 726],
            [485, 753],
            [485, 728],
            [492, 762],
            [495, 745],
            [491, 910],
            [493, 909],
            [499, 904],
            [505, 905],
            [504, 747],
            [515, 743],
            [516, 752],
            [524, 855],
            [525, 844],
            [525, 885],
            [533, 845],
            [533, 873],
            [535, 883],
            [545, 874],
            [543, 864],
            [553, 865],
            [553, 845],
            [554, 825],
            [554, 835],
            [563, 845],
            [565, 826],
            [563, 855],
            [563, 795],
            [565, 735],
            [573, 778],
            [572, 815],
            [574, 804],
            [575, 665],
            [575, 685],
            [574, 705],
            [574, 745],
            [575, 875],
            [572, 732],
            [582, 795],
            [579, 709],
            [583, 805],
            [583, 854],
            [586, 755],
            [584, 824],
            [585, 655],
            [581, 718],
            [586, 844],
            [585, 915],
            [587, 905],
            [594, 824],
            [593, 855],
            [590, 891],
            [594, 776],
            [596, 767],
            [593, 763],
            [603, 785],
            [604, 775],
            [603, 885],
            [605, 753],
            [605, 655],
            [606, 935],
            [603, 761],
            [613, 802],
            [613, 945],
            [613, 965],
            [615, 693],
            [617, 665],
            [623, 962],
            [624, 972],
            [625, 995],
            [633, 673],
            [633, 965],
            [633, 683],
            [633, 692],
            [633, 954],
            [634, 1016],
            [635, 664],
            [641, 804],
            [637, 999],
            [641, 956],
            [643, 946],
            [643, 926],
            [644, 975],
            [643, 655],
            [646, 705],
            [651, 664],
            [651, 984],
            [647, 665],
            [651, 715],
            [651, 725],
            [651, 734],
            [647, 809],
            [651, 825],
            [651, 873],
            [647, 900],
            [652, 917],
            [651, 944],
            [652, 742],
            [648, 811],
            [651, 994],
            [652, 783],
            [650, 911],
            [654, 879],
        ],
        dtype=np.int32,
    )

    # estimate parameters of real data
    model = EllipseModel.from_estimate(data)

    # test whether estimated parameters are smaller then 1000, so means stable
    assert np.all(model.center < 1000)
    assert np.all(model.axis_lengths < 1000)

    # test whether all parameters are more than 0. Negative values were the
    # result of an integer overflow
    assert np.all(model.center > 0)
    assert np.all(model.axis_lengths > 0)

    # estimate method
    model2 = EllipseModel(*DUMMY_ELLIPSE_ARGS)
    with pytest.warns(FutureWarning, match='`estimate` is deprecated') as record:
        assert model2.estimate(data)
    assert_stacklevel(record)
    assert len(record) == 1
    _assert_ellipse_equal(model, model2)


def test_ellipse_model_estimate_from_far_shifted_data():
    params = np.array([1e6, 2e6, 0.5, 0.1, 0.5], dtype=np.float64)
    center, a, b, theta = params[:2], params[2], params[3], params[4]
    angles = np.array(
        [
            0.107,
            0.407,
            1.108,
            1.489,
            2.216,
            2.768,
            3.183,
            3.969,
            4.840,
            5.387,
            5.792,
            6.139,
        ],
        dtype=np.float64,
    )
    em = EllipseModel(center, (a, b), theta)
    data = em.predict_xy(angles)
    # assert that far shifted data can be estimated
    float_data = data.astype(np.float64)
    model = EllipseModel.from_estimate(float_data)
    # test whether the predicted parameters are close to the original ones
    _assert_ellipse_almost_equal(em, model)
    model2 = EllipseModel(*DUMMY_ELLIPSE_ARGS)
    with pytest.warns(FutureWarning, match='`estimate` is deprecated') as record:
        assert model2.estimate(float_data)
    assert_stacklevel(record)
    assert len(record) == 1
    _assert_ellipse_almost_equal(em, model2)


# Passing on WASM
@xfail(
    condition=arch32 and not is_wasm,
    reason=(
        'Known test failure on 32-bit platforms. See links for '
        'details: '
        'https://github.com/scikit-image/scikit-image/issues/3091 '
        'https://github.com/scikit-image/scikit-image/issues/2670'
    ),
)
@pytest.mark.parametrize(
    ("data", "msg"),
    [
        (
            np.ones((6, 2)),
            "Standard deviation of data is too small to estimate "
            "ellipse with meaningful precision.",
        ),
        (
            np.array([[50, 80], [51, 81], [52, 80]]),
            "Need at least 5 data points to estimate an ellipse.",
        ),
    ],
)
def test_ellipse_model_estimate_failers(data, msg):
    # estimate parameters of real data
    dep_msg = '`estimate` is deprecated'

    tf = EllipseModel.from_estimate(data)
    assert not tf
    assert str(tf).endswith(msg)

    tf = EllipseModel(*DUMMY_ELLIPSE_ARGS)
    with pytest.warns(FutureWarning, match=dep_msg):
        with pytest.warns(RuntimeWarning, match=msg) as _warnings:
            assert tf.estimate(data)
    assert_stacklevel(_warnings)
    assert len(_warnings) == 2


def test_ellipse_model_residuals():
    # vertical line through origin
    model = EllipseModel((0, 0), (10, 5), 0)
    assert_almost_equal(abs(model.residuals(np.array([[10, 0]]))), 0)
    assert_almost_equal(abs(model.residuals(np.array([[0, 5]]))), 0)
    assert_almost_equal(abs(model.residuals(np.array([[0, 10]]))), 5)


def test_ransac_shape():
    # generate original data without noise
    model0 = CircleModel((10, 12), 3)
    t = np.linspace(0, 2 * np.pi, 1000)
    data0 = model0.predict_xy(t)

    # add some faulty data
    outliers = (10, 30, 200)
    data0[outliers[0], :] = (1000, 1000)
    data0[outliers[1], :] = (-50, 50)
    data0[outliers[2], :] = (-100, -10)

    # estimate parameters of corrupted data
    model_est, inliers = ransac(data0, CircleModel, 3, 5, rng=1)
    ransac(data0, CircleModel, 3, 5, rng=1)

    # test whether estimated parameters equal original parameters
    assert_almost_equal(model0.center, model_est.center)
    assert_almost_equal(model0.radius, model_est.radius)
    for outlier in outliers:
        assert outlier not in inliers


@pytest.fixture
def ransac_params():
    rng = np.random.default_rng(12373240)

    # generate original data without noise
    src = 100 * rng.random((50, 2))
    model = AffineTransform(scale=(0.5, 0.3), rotation=1, translation=(10, 20))
    dst = model(src)

    # add some faulty data
    outliers = (0, 5, 20)
    dst[outliers[0]] = (10000, 10000)
    dst[outliers[1]] = (-100, 100)
    dst[outliers[2]] = (50, 50)
    return src, dst, model, outliers, rng


def test_ransac_geometric(ransac_params):
    src, dst, model0, outliers, rng = ransac_params
    # estimate parameters of corrupted data
    model_est, inliers = ransac((src, dst), AffineTransform, 2, 20, rng=rng)

    # test whether estimated parameters equal original parameters
    assert_almost_equal(model0.params, model_est.params)
    assert np.all(np.nonzero(inliers == False)[0] == outliers)


def test_custom_estimate_warning(ransac_params):
    # Test that custom estimate class raises warning.
    src, dst, model0, outliers, rng = ransac_params

    class C:
        """Custom class"""

        def __init__(self):
            self._model = AffineTransform()

        @property
        def params(self):
            return self._model.params

        def estimate(self, src, dst):
            self._model = AffineTransform.from_estimate(src, dst)
            return bool(self._model)

        def residuals(self, src, dst):
            return self._model.residuals(src, dst)

    msg = (
        "Passing custom classes without `from_estimate` has been deprecated "
        "since version 0.26 and will be removed in version 2.2. "
        "Add `from_estimate` class method to custom class to avoid this "
        "warning."
    )
    with pytest.warns(FutureWarning, match=msg) as record:
        model_est, inliers = ransac((src, dst), C, 2, 20, rng=rng)
    assert_stacklevel(record)
    assert len(record) == 1

    assert_almost_equal(model0.params, model_est.params)

    # Test modified class maatches standard from_estimate behavior.
    with pytest.warns(FutureWarning, match=msg) as record:
        patched_class = add_from_estimate(C)

    tf = patched_class.from_estimate(src, dst)
    assert bool(tf)
    bad_tf = patched_class.from_estimate(np.ones_like(src), dst)
    assert not bool(bad_tf)
    assert str(bad_tf) == '`C` estimation failed'


def test_ransac_model_class_protocol(ransac_params):
    # Test custom classes that don't match protocol.
    src, dst, model0, outliers, rng = ransac_params

    class D:
        """Class without `residuals` method."""

        @classmethod
        def from_estimate(cls, data):
            return cls()

    with pytest.raises(TypeError, match='`model_class` '):
        ransac((src, dst), D, 2, 20, rng=rng)

    class E:
        """Class without `from_estimate` or `estimate`"""

        def residuals(self, data):
            return data

    with pytest.raises(TypeError, match='Class .* must have `from_estimate` '):
        ransac((src, dst), E, 2, 20, rng=rng)


def test_custom_from_estimate_classmethod(ransac_params):
    # Test assertion that custom class `from_estimate` is class method.
    src, dst, model0, outliers, rng = ransac_params

    class F:
        """Class without `from_estimate` or `estimate`"""

        def from_estimate(self, data):
            return self

        def residuals(self, data):
            return data

    with pytest.raises(TypeError, match='`from_estimate` must be a class method'):
        ransac((src, dst), F, 2, 20, rng=rng)


def test_ransac_is_data_valid():
    def is_data_valid(data):
        return data.shape[0] > 2

    with expected_warnings(["No inliers found"]):
        model, inliers = ransac(
            np.empty((10, 2)),
            LineModelND,
            2,
            np.inf,
            is_data_valid=is_data_valid,
            rng=1,
        )
    assert_equal(model, None)
    assert_equal(inliers, None)


def test_ransac_is_model_valid():
    def is_model_valid(model, data):
        return False

    with pytest.warns(UserWarning, match="No inliers found"):
        model, inliers = ransac(
            np.zeros((10, 2), dtype=np.float64),
            LineModelND,
            2,
            np.inf,
            is_model_valid=is_model_valid,
            rng=1,
        )
    assert_equal(model, None)
    assert_equal(inliers, None)


def test_ransac_dynamic_max_trials():
    # Numbers hand-calculated and confirmed on page 119 (Table 4.3) in
    #   Hartley, R.~I. and Zisserman, A., 2004,
    #   Multiple View Geometry in Computer Vision, Second Edition,
    #   Cambridge University Press, ISBN: 0521540518

    # e = 0%, min_samples = X
    assert_equal(_dynamic_max_trials(100, 100, 2, 0.99), 1)
    assert_equal(_dynamic_max_trials(100, 100, 2, 1), 1)

    # e = 5%, min_samples = 2
    assert_equal(_dynamic_max_trials(95, 100, 2, 0.99), 2)
    assert_equal(_dynamic_max_trials(95, 100, 2, 1), 16)
    # e = 10%, min_samples = 2
    assert_equal(_dynamic_max_trials(90, 100, 2, 0.99), 3)
    assert_equal(_dynamic_max_trials(90, 100, 2, 1), 22)
    # e = 30%, min_samples = 2
    assert_equal(_dynamic_max_trials(70, 100, 2, 0.99), 7)
    assert_equal(_dynamic_max_trials(70, 100, 2, 1), 54)
    # e = 50%, min_samples = 2
    assert_equal(_dynamic_max_trials(50, 100, 2, 0.99), 17)
    assert_equal(_dynamic_max_trials(50, 100, 2, 1), 126)

    # e = 5%, min_samples = 8
    assert_equal(_dynamic_max_trials(95, 100, 8, 0.99), 5)
    assert_equal(_dynamic_max_trials(95, 100, 8, 1), 34)
    # e = 10%, min_samples = 8
    assert_equal(_dynamic_max_trials(90, 100, 8, 0.99), 9)
    assert_equal(_dynamic_max_trials(90, 100, 8, 1), 65)
    # e = 30%, min_samples = 8
    assert_equal(_dynamic_max_trials(70, 100, 8, 0.99), 78)
    assert_equal(_dynamic_max_trials(70, 100, 8, 1), 608)
    # e = 50%, min_samples = 8
    assert_equal(_dynamic_max_trials(50, 100, 8, 0.99), 1177)
    assert_equal(_dynamic_max_trials(50, 100, 8, 1), 9210)

    # e = 0%, min_samples = 5
    assert_equal(_dynamic_max_trials(1, 100, 5, 0), 0)
    assert_equal(_dynamic_max_trials(1, 100, 5, 1), 360436504051)


def test_ransac_dynamic_max_trials_clipping():
    """Test that the function behaves well when `nom` or `denom` become almost 1.0."""
    # e = 0%, min_samples = 10
    # Ensure that (1 - inlier_ratio ** min_samples) approx 1 does not fail.
    assert_equal(_dynamic_max_trials(1, 100, 10, 0), 0)

    EPSILON = np.finfo(np.float64).eps
    desired = np.ceil(np.log(EPSILON) / np.log(1 - EPSILON))
    assert desired > 0
    assert_equal(_dynamic_max_trials(1, 100, 1000, 1), desired)

    # Ensure that (1 - probability) approx 1 does not fail.
    assert_equal(_dynamic_max_trials(1, 100, 10, 1e-40), 1)
    assert_equal(_dynamic_max_trials(1, 100, 1000, 1e-40), 1)


def test_ransac_invalid_input():
    # `residual_threshold` must be greater than zero
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=2, residual_threshold=-0.5)
    # "`max_trials` must be greater than zero"
    with testing.raises(ValueError):
        ransac(
            np.zeros((10, 2)), None, min_samples=2, residual_threshold=0, max_trials=-1
        )
    # `stop_probability` must be in range (0, 1)
    with testing.raises(ValueError):
        ransac(
            np.zeros((10, 2)),
            None,
            min_samples=2,
            residual_threshold=0,
            stop_probability=-1,
        )
    # `stop_probability` must be in range (0, 1)
    with testing.raises(ValueError):
        ransac(
            np.zeros((10, 2)),
            None,
            min_samples=2,
            residual_threshold=0,
            stop_probability=1.01,
        )
    # `min_samples` as ratio must be in range (0, nb)
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=0, residual_threshold=0)
    # `min_samples` as ratio must be in range (0, nb]
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=11, residual_threshold=0)
    # `min_samples` must be greater than zero
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=-1, residual_threshold=0)


def test_ransac_sample_duplicates():
    class DummyModel:
        """Dummy model to check for duplicates."""

        @classmethod
        def from_estimate(cls, data):
            # Assert that all data points are unique.
            assert_equal(np.unique(data).size, data.size)
            return cls()

        def residuals(self, data):
            return np.ones(len(data), dtype=np.float64)

    # Create dataset with four unique points. Force 10 iterations
    # and check that there are no duplicated data points.
    data = np.arange(4)
    with pytest.warns(UserWarning, match="No inliers found"):
        ransac(data, DummyModel, min_samples=3, residual_threshold=0.0, max_trials=10)


def test_ransac_with_no_final_inliers():
    data = np.random.rand(5, 2)
    with pytest.warns(UserWarning, match='No inliers found. Model not fitted'):
        model, inliers = ransac(
            data,
            model_class=LineModelND,
            min_samples=3,
            residual_threshold=0,
            rng=1523427,
        )
    assert inliers is None
    assert model is None


def test_ransac_non_valid_best_model():
    """Example from GH issue #5572"""

    def is_model_valid(model, *random_data) -> bool:
        """Allow models with a maximum of 10 degree tilt from the vertical"""
        tilt = abs(np.arccos(np.dot(model.direction, [0, 0, 1])))
        return tilt <= (10 / 180 * np.pi)

    rng = np.random.RandomState(1)
    data = np.linspace([0, 0, 0], [0.3, 0, 1], 1000) + rng.rand(1000, 3) - 0.5
    with pytest.warns(UserWarning, match="Estimated model is not valid"):
        ransac(
            data,
            LineModelND,
            min_samples=2,
            residual_threshold=0.3,
            max_trials=50,
            rng=0,
            is_model_valid=is_model_valid,
        )


@pytest.mark.parametrize('tf_class', MODEL_CLASSES)
def test_init_estimate_deprecations(tf_class):
    rng = np.random.default_rng()
    data = rng.normal(100, 40, size=(10, 2))
    assert tf_class.from_estimate(data)
    class_name = tf_class.__name__
    init_msg = (
        rf'Calling ``{class_name}\(\)`` \(without arguments\) has been '
        'deprecated since version 0.26 and will be removed in version 2.2; '
        f'see help for ``{class_name}``.'
    )
    with pytest.warns(FutureWarning, match=init_msg) as record:
        tf = tf_class()
    assert_stacklevel(record)
    assert len(record) == 1
    estimate_msg = (
        f'`estimate` is deprecated since .* Please use `{tf_class.__name__}'
        '.from_estimate` class constructor instead.'
    )
    with pytest.warns(FutureWarning, match=estimate_msg) as record:
        assert tf.estimate(data)
    assert_stacklevel(record)
    assert len(record) == 1


@pytest.mark.parametrize(
    'tf_class, dummy_args',
    (
        (LineModelND, DUMMY_LINE_ARGS),
        (CircleModel, DUMMY_CIRCLE_ARGS),
        (EllipseModel, DUMMY_ELLIPSE_ARGS),
    ),
)
def test_params(tf_class, dummy_args):
    tf = tf_class(*dummy_args)
    msg = '`params` attribute deprecated; use .* attributes instead'
    with pytest.warns(FutureWarning, match=msg) as record:
        p = tf.params
    assert_stacklevel(record)
    if tf_class is LineModelND:
        assert np.all(p[0] == tf.origin)
        assert np.all(p[1] == tf.direction)
    else:
        assert np.all(p == np.r_[*dummy_args])
