import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
from skimage.filters._lic import line_integral_convolution as lic
from skimage._shared._warnings import expected_warnings


def test_lic_1D():
    image = np.zeros((5,))
    image[3] = 2
    ones_ = np.ones_like(image)
    velocity = 10 * ones_[:, None]
    kernel = np.array([1, 0.5, 0.1])
    assert_allclose(lic(image, velocity, kernel, order=1, origin=-1,
                        weighted='integral'), [0, 0.2, 1, 2, 0])
    assert_allclose(lic(image, velocity, kernel, order=1, origin=None,
                        weighted='integral'), [0, 0.2, 1, 2, 0])
    assert_allclose(lic(ones_, velocity, kernel, order=1, weighted='average'),
                    ones_)

    assert_allclose(lic(image, velocity, kernel, order=1, weighted='integral'),
                    [0, 0, 0.2, 1, 2])

    assert_allclose(lic(image.astype(bool), velocity, kernel, order=1,
                        weighted='integral'), [0, 0, 0.1, 0.5, 1])

    assert_equal(lic(image.astype(bool), velocity, kernel, order=1,
                     weighted='integral').dtype.type,
                 np.float64)

    assert_equal(lic(image.astype(bool), velocity.astype(np.float32),
                     kernel.astype(np.float32),
                     order=1, weighted='integral').dtype.type,
                 np.float32)

    assert_equal(lic(image.astype(bool), velocity,
                     kernel.astype(np.float32),
                     order=1, weighted='integral').dtype.type,
                 np.float64)

    assert_allclose(lic(image, velocity, kernel, order=1, origin=None,
                        weighted='integral', step_size='unit_time',
                        maximum_velocity=1.), [0, 0.2, 1, 2, 0])

    assert_allclose(lic(image, velocity, kernel, order=1, origin=None,
                        weighted='integral', step_size='unit_time',
                        maximum_velocity=0.5), [0, 0, 0.35, 1.25, 0])


def test_lic_2D():
    # was used to generate results for test cases
    # np.set_printoptions(
    #     formatter={'float':lambda x:'%.15e' % x},
    #     threshold=np.nan,
    #     linewidth=200)
    image = np.zeros((4, 5))
    image[1, 3] = 2
    ones_ = np.ones_like(image)
    velocity = ones_[:, :, None] * [[[0, 2]]]
    kernel = [1, 0.5, 0.1]
    assert_allclose(lic(image, velocity, kernel, order=1, origin=-1,
                        weighted='integral'),
                    [[0, 0, 0, 0, 0],
                     [0, 0.2, 1, 2, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])

    velocity = ones_[:, :, None] * [[[2, 0]]]
    assert_allclose(lic(image, velocity, kernel, order=1, origin=-1,
                        weighted='integral'),
                    [[0, 0, 0, 1, 0],
                     [0, 0, 0, 2, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])

    velocity = (ones_[:, :, None] * [[[2, 7]]])
    assert_allclose(lic(image, velocity, kernel, order=1, origin=-1,
                        weighted='integral'),
                    [[0, 1.014323035580300e-01, 2.726070909971477e-01,
                      1.057018450115160e-02, 0],
                     [0, 8.317727549829940e-02, 7.043072775873462e-01,
                        2.027905867858025e+00, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])
    assert_allclose(lic(image, velocity, kernel, order=3, origin=-1,
                        weighted='integral'),
                    [[-4.946647656636081e-03, 1.041909046074850e-01,
                      1.765005843082361e-01, 6.998764796487109e-04, 0],
                     [3.736874054193710e-04, 8.431107873364971e-02,
                        8.935011111472939e-01, 2.003570754474494e+00, 0],
                     [1.425762748747135e-04, -1.431941717202221e-02,
                        -1.164110107604186e-01, -4.650116996736128e-04, 0],
                     [0, 0, 0, 0, 0]])


if __name__ == "__main__":
    from numpy import testing
    testing.run_module_suite()
