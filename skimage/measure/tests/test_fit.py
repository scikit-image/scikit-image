import numpy as np
from scipy import spatial
from numpy.testing import assert_equal, assert_raises, assert_almost_equal
from skimage.measure import LineModelND, CircleModel, EllipseModel, ransac
from skimage.transform import AffineTransform
from skimage.measure.fit import _dynamic_max_trials
from skimage._shared._warnings import expected_warnings


def test_line_model_invalid_input():
    assert_raises(ValueError, LineModelND().estimate, np.empty((1, 3)))


def test_line_model_predict():
    model = LineModelND()
    model.params = ((0, 0), (1, 1))
    x = np.arange(-10, 10)
    y = model.predict_y(x)
    assert_almost_equal(x, model.predict_x(y))


def test_line_model_estimate():
    # generate original data without noise
    model0 = LineModelND()
    model0.params = ((0, 0), (1, 1))
    x0 = np.arange(-100, 100)
    y0 = model0.predict_y(x0)

    data = np.column_stack([x0, y0])

    # estimate parameters of noisy data
    model_est = LineModelND()
    model_est.estimate(data)

    # test whether estimated parameters almost equal original parameters
    random_state = np.random.RandomState(1234)
    x = random_state.rand(100, 2)
    assert_almost_equal(model0.predict(x), model_est.predict(x), 1)


def test_line_model_residuals():
    model = LineModelND()
    model.params = (np.array([0, 0]), np.array([0, 1]))
    assert_equal(model.residuals(np.array([[0, 0]])), 0)
    assert_equal(model.residuals(np.array([[0, 10]])), 0)
    assert_equal(model.residuals(np.array([[10, 0]])), 10)
    model.params = (np.array([-2, 0]), np.array([1, 1])  / np.sqrt(2))
    assert_equal(model.residuals(np.array([[0, 0]])), np.sqrt(2))
    assert_almost_equal(model.residuals(np.array([[-4, 0]])), np.sqrt(2))


def test_line_model_under_determined():
    data = np.empty((1, 2))
    assert_raises(ValueError, LineModelND().estimate, data)


def test_line_modelND_invalid_input():
    assert_raises(ValueError, LineModelND().estimate, np.empty((5, 1)))


def test_line_modelND_predict():
    model = LineModelND()
    model.params = (np.array([0, 0]), np.array([0.2, 0.98]))
    x = np.arange(-10, 10)
    y = model.predict_y(x)
    assert_almost_equal(x, model.predict_x(y))


def test_line_modelND_estimate():
    # generate original data without noise
    model0 = LineModelND()
    model0.params = (np.array([0,0,0], dtype='float'),
                         np.array([1,1,1], dtype='float')/np.sqrt(3))
    # we scale the unit vector with a factor 10 when generating points on the
    # line in order to compensate for the scale of the random noise
    data0 = (model0.params[0] +
             10 * np.arange(-100,100)[...,np.newaxis] * model0.params[1])

    # add gaussian noise to data
    random_state = np.random.RandomState(1234)
    data = data0 + random_state.normal(size=data0.shape)

    # estimate parameters of noisy data
    model_est = LineModelND()
    model_est.estimate(data)

    # test whether estimated parameters are correct
    # we use the following geometric property: two aligned vectors have
    # a cross-product equal to zero
    # test if direction vectors are aligned
    assert_almost_equal(np.linalg.norm(np.cross(model0.params[1],
                                                model_est.params[1])), 0, 1)
    # test if origins are aligned with the direction
    a = model_est.params[0] - model0.params[0]
    if np.linalg.norm(a) > 0:
        a /= np.linalg.norm(a)
    assert_almost_equal(np.linalg.norm(np.cross(model0.params[1], a)), 0, 1)


def test_line_modelND_residuals():
    model = LineModelND()
    model.params = (np.array([0, 0, 0]), np.array([0, 0, 1]))
    assert_equal(abs(model.residuals(np.array([[0, 0, 0]]))), 0)
    assert_equal(abs(model.residuals(np.array([[0, 0, 1]]))), 0)
    assert_equal(abs(model.residuals(np.array([[10, 0, 0]]))), 10)


def test_line_modelND_under_determined():
    data = np.empty((1, 3))
    assert_raises(ValueError, LineModelND().estimate, data)


def test_circle_model_invalid_input():
    assert_raises(ValueError, CircleModel().estimate, np.empty((5, 3)))
    assert_raises(ValueError, CircleModel().estimate, np.zeros((5, 2)),
                  init_params=(0., 0.))


def test_circle_model_predict():
    model = CircleModel()
    r = 5
    model.params = (0, 0, r)
    t = np.arange(0, 2 * np.pi, np.pi / 2)

    xy = np.array(((5, 0), (0, 5), (-5, 0), (0, -5)))
    assert_almost_equal(xy, model.predict_xy(t))


def test_circle_model_estimate():
    # generate original data without noise
    model0 = CircleModel()
    model0.params = (10, 12, 3)
    t = np.linspace(0, 2 * np.pi, 1000)
    data0 = model0.predict_xy(t)

    # add gaussian noise to data
    random_state = np.random.RandomState(1234)
    data = data0 + random_state.normal(size=data0.shape)

    # estimate parameters of noisy data
    model_est = CircleModel()
    model_est.estimate(data)

    # test whether estimated parameters almost equal original parameters
    assert_almost_equal(model0.params, model_est.params, 1)


def test_circle_model_residuals():
    model = CircleModel()
    model.params = (0, 0, 5)
    assert_almost_equal(abs(model.residuals(np.array([[5, 0]]))), 0)
    assert_almost_equal(abs(model.residuals(np.array([[6, 6]]))),
                        np.sqrt(2 * 6**2) - 5)
    assert_almost_equal(abs(model.residuals(np.array([[10, 0]]))), 5)


def test_ellipse_model_invalid_input():
    assert_raises(ValueError, EllipseModel().estimate, np.empty((5, 3)))
    assert_raises(ValueError, EllipseModel().estimate, np.zeros((5, 2)),
                  init_params=(0, 0, 0))


def test_ellipse_model_predict():
    model = EllipseModel()
    r = 5
    model.params = (0, 0, 5, 10, 0)
    t = np.arange(0, 2 * np.pi, np.pi / 2)

    xy = np.array(((5, 0), (0, 10), (-5, 0), (0, -10)))
    assert_almost_equal(xy, model.predict_xy(t))


def test_ellipse_model_estimate():
    # generate original data without noise
    model0 = EllipseModel()
    model0.params = (10, 20, 15, 25, 0)
    t = np.linspace(0, 2 * np.pi, 100)
    data0 = model0.predict_xy(t)

    # add gaussian noise to data
    random_state = np.random.RandomState(1234)
    data = data0 + random_state.normal(size=data0.shape)

    # estimate parameters of noisy data
    model_est = EllipseModel()
    model_est.estimate(data)

    # test whether estimated parameters almost equal original parameters
    assert_almost_equal(model0.params, model_est.params, 0)


def test_ellipse_model_residuals():
    model = EllipseModel()
    # vertical line through origin
    model.params = (0, 0, 10, 5, 0)
    assert_almost_equal(abs(model.residuals(np.array([[10, 0]]))), 0)
    assert_almost_equal(abs(model.residuals(np.array([[0, 5]]))), 0)
    assert_almost_equal(abs(model.residuals(np.array([[0, 10]]))), 5)


def test_ransac_shape():
    # generate original data without noise
    model0 = CircleModel()
    model0.params = (10, 12, 3)
    t = np.linspace(0, 2 * np.pi, 1000)
    data0 = model0.predict_xy(t)

    # add some faulty data
    outliers = (10, 30, 200)
    data0[outliers[0], :] = (1000, 1000)
    data0[outliers[1], :] = (-50, 50)
    data0[outliers[2], :] = (-100, -10)

    # estimate parameters of corrupted data
    model_est, inliers = ransac(data0, CircleModel, 3, 5,
                                random_state=1)

    # test whether estimated parameters equal original parameters
    assert_equal(model0.params, model_est.params)
    for outlier in outliers:
        assert outlier not in inliers


def test_ransac_geometric():
    random_state = np.random.RandomState(1)

    # generate original data without noise
    src = 100 * random_state.random_sample((50, 2))
    model0 = AffineTransform(scale=(0.5, 0.3), rotation=1,
                             translation=(10, 20))
    dst = model0(src)

    # add some faulty data
    outliers = (0, 5, 20)
    dst[outliers[0]] = (10000, 10000)
    dst[outliers[1]] = (-100, 100)
    dst[outliers[2]] = (50, 50)

    # estimate parameters of corrupted data
    model_est, inliers = ransac((src, dst), AffineTransform, 2, 20,
                                random_state=random_state)

    # test whether estimated parameters equal original parameters
    assert_almost_equal(model0.params, model_est.params)
    assert np.all(np.nonzero(inliers == False)[0] == outliers)


def test_ransac_is_data_valid():

    is_data_valid = lambda data: data.shape[0] > 2
    model, inliers = ransac(np.empty((10, 2)), LineModelND, 2, np.inf,
                            is_data_valid=is_data_valid, random_state=1)
    assert_equal(model, None)
    assert_equal(inliers, None)


def test_ransac_is_model_valid():

    def is_model_valid(model, data):
        return False
    model, inliers = ransac(np.empty((10, 2)), LineModelND, 2, np.inf,
                            is_model_valid=is_model_valid, random_state=1)
    assert_equal(model, None)
    assert_equal(inliers, None)


def test_ransac_dynamic_max_trials():
    # Numbers hand-calculated and confirmed on page 119 (Table 4.3) in
    #   Hartley, R.~I. and Zisserman, A., 2004,
    #   Multiple View Geometry in Computer Vision, Second Edition,
    #   Cambridge University Press, ISBN: 0521540518

    # e = 0%, min_samples = X
    assert_equal(_dynamic_max_trials(100, 100, 2, 0.99), 1)

    # e = 5%, min_samples = 2
    assert_equal(_dynamic_max_trials(95, 100, 2, 0.99), 2)
    # e = 10%, min_samples = 2
    assert_equal(_dynamic_max_trials(90, 100, 2, 0.99), 3)
    # e = 30%, min_samples = 2
    assert_equal(_dynamic_max_trials(70, 100, 2, 0.99), 7)
    # e = 50%, min_samples = 2
    assert_equal(_dynamic_max_trials(50, 100, 2, 0.99), 17)

    # e = 5%, min_samples = 8
    assert_equal(_dynamic_max_trials(95, 100, 8, 0.99), 5)
    # e = 10%, min_samples = 8
    assert_equal(_dynamic_max_trials(90, 100, 8, 0.99), 9)
    # e = 30%, min_samples = 8
    assert_equal(_dynamic_max_trials(70, 100, 8, 0.99), 78)
    # e = 50%, min_samples = 8
    assert_equal(_dynamic_max_trials(50, 100, 8, 0.99), 1177)

    # e = 0%, min_samples = 5
    assert_equal(_dynamic_max_trials(1, 100, 5, 0), 0)
    assert_equal(_dynamic_max_trials(1, 100, 5, 1), np.inf)


def test_ransac_invalid_input():
    assert_raises(ValueError, ransac, np.zeros((10, 2)), None, min_samples=2,
                  residual_threshold=0, max_trials=-1)
    assert_raises(ValueError, ransac, np.zeros((10, 2)), None, min_samples=2,
                  residual_threshold=0, stop_probability=-1)
    assert_raises(ValueError, ransac, np.zeros((10, 2)), None, min_samples=2,
                  residual_threshold=0, stop_probability=1.01)
    assert_raises(ValueError, ransac, np.zeros((10, 2)), None, min_samples=0,
                  residual_threshold=0)
    assert_raises(ValueError, ransac, np.zeros((10, 2)), None, min_samples=1.5,
                  residual_threshold=0)


def test_ransac_initial_guess():
    center = [242., 697.]
    points = np.array(
        [[105, 835], [105, 845], [105, 865], [105, 875], [107, 855], [115, 825],
         [115, 835], [116, 845], [115, 865], [115, 885], [117, 855], [125, 825],
         [125, 835], [125, 895], [135, 818], [134, 893], [137, 888], [144, 832],
         [146, 825], [144, 895], [141, 826], [155, 897], [155, 675], [156, 695],
         [155, 825], [157, 685], [153, 831], [163, 726], [164, 824], [165, 902],
         [165, 645], [166, 655], [165, 665], [164, 705], [164, 715], [166, 735],
         [166, 745], [166, 674], [165, 695], [167, 684], [173, 724], [173, 705],
         [175, 715], [175, 818], [175, 635], [175, 645], [174, 755], [175, 902],
         [176, 655], [177, 665], [176, 735], [177, 745], [177, 765], [185, 754],
         [184, 885], [184, 635], [186, 822], [186, 895], [185, 774], [182, 893],
         [188, 763], [194, 884], [193, 777], [196, 768], [195, 785], [195, 817],
         [192, 881], [203, 824], [205, 876], [204, 777], [203, 885], [200, 781],
         [204, 788], [214, 783], [215, 794], [215, 892], [216, 816], [221, 876],
         [223, 825], [225, 886], [225, 784], [225, 795], [225, 805], [227, 834],
         [234, 793], [230, 880], [235, 883], [245, 875], [245, 605], [245, 801],
         [240, 799], [244, 814], [247, 826], [255, 796], [253, 864], [255, 605],
         [251, 860], [265, 607], [274, 606], [275, 596], [285, 597], [284, 607],
         [295, 605], [294, 616], [295, 625], [297, 764], [304, 615], [303, 635],
         [305, 625], [305, 647], [306, 656], [307, 715], [306, 744], [307, 762],
         [313, 635], [313, 645], [312, 734], [314, 756], [316, 767], [312, 665],
         [313, 684], [313, 695], [317, 654], [314, 705], [315, 745], [315, 725],
         [318, 674], [317, 715], [323, 665], [323, 695], [322, 735], [323, 685],
         [325, 705], [325, 763], [325, 725], [327, 675]])

    ransac_model, inliers = ransac(points, EllipseModel, min_samples=0.4,
                                   residual_threshold=15, max_trials=50)
    nb_inliers_no_init = np.sum(inliers)

    dists = spatial.distance.cdist(points, [center], metric='euclidean')
    init_inliers = (dists <= np.mean(dists))[:, 0]
    ransac_model, inliers = ransac(points, EllipseModel, min_samples=0.4,
                                   residual_threshold=15, max_trials=50,
                                   init_inliers=init_inliers)
    nb_inliers_w_init = np.sum(inliers)
    assert  nb_inliers_no_init < nb_inliers_w_init


def test_ransac_initial_guess():
    center = [242., 697.]
    points = np.array(
        [[105, 835], [105, 845], [105, 865], [105, 875], [107, 855], [115, 825],
         [115, 835], [116, 845], [115, 865], [115, 885], [117, 855], [125, 825],
         [125, 835], [125, 895], [135, 818], [134, 893], [137, 888], [144, 832],
         [146, 825], [144, 895], [141, 826], [155, 897], [155, 675], [156, 695],
         [155, 825], [157, 685], [153, 831], [163, 726], [164, 824], [165, 902],
         [165, 645], [166, 655], [165, 665], [164, 705], [164, 715], [166, 735],
         [166, 745], [166, 674], [165, 695], [167, 684], [173, 724], [173, 705],
         [175, 715], [175, 818], [175, 635], [175, 645], [174, 755], [175, 902],
         [176, 655], [177, 665], [176, 735], [177, 745], [177, 765], [185, 754],
         [184, 885], [184, 635], [186, 822], [186, 895], [185, 774], [182, 893],
         [188, 763], [194, 884], [193, 777], [196, 768], [195, 785], [195, 817],
         [192, 881], [203, 824], [205, 876], [204, 777], [203, 885], [200, 781],
         [204, 788], [214, 783], [215, 794], [215, 892], [216, 816], [221, 876],
         [223, 825], [225, 886], [225, 784], [225, 795], [225, 805], [227, 834],
         [234, 793], [230, 880], [235, 883], [245, 875], [245, 605], [245, 801],
         [240, 799], [244, 814], [247, 826], [255, 796], [253, 864], [255, 605],
         [251, 860], [265, 607], [274, 606], [275, 596], [285, 597], [284, 607],
         [295, 605], [294, 616], [295, 625], [297, 764], [304, 615], [303, 635],
         [305, 625], [305, 647], [306, 656], [307, 715], [306, 744], [307, 762],
         [313, 635], [313, 645], [312, 734], [314, 756], [316, 767], [312, 665],
         [313, 684], [313, 695], [317, 654], [314, 705], [315, 745], [315, 725],
         [318, 674], [317, 715], [323, 665], [323, 695], [322, 735], [323, 685],
         [325, 705], [325, 763], [325, 725], [327, 675]])

    ransac_model, inliers = ransac(points, EllipseModel, min_samples=0.4,
                                   residual_threshold=15, max_trials=50)
    nb_inliers_no_init = np.sum(inliers)

    dists = spatial.distance.cdist(points, [center], metric='euclidean')
    init_inliers = (dists <= np.mean(dists))[:, 0]
    ransac_model, inliers = ransac(points, EllipseModel, min_samples=0.4,
                                   residual_threshold=15, max_trials=50,
                                   init_samples=init_inliers)
    nb_inliers_w_init = np.sum(inliers)
    assert  nb_inliers_no_init < nb_inliers_w_init


if __name__ == "__main__":
    np.testing.run_module_suite()
