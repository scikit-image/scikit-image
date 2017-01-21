import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest

from skimage.draw import ellipsoid, ellipsoid_stats


def test_ellipsoid_sign_parameters1():
    with pytest.raises(ValueError):
        ellipsoid(-1, 2, 2)


def test_ellipsoid_sign_parameters2():
    with pytest.raises(ValueError):
        ellipsoid(0, 2, 2)


def test_ellipsoid_sign_parameters3():
    with pytest.raises(ValueError):
        ellipsoid(-3, -2, 2)


def test_ellipsoid_bool():
    test = ellipsoid(2, 2, 2)[1:-1, 1:-1, 1:-1]
    test_anisotropic = ellipsoid(2, 2, 4, spacing=(1., 1., 2.))
    test_anisotropic = test_anisotropic[1:-1, 1:-1, 1:-1]

    expected = np.array([[[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]],

                         [[0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0]],

                         [[0, 0, 1, 0, 0],
                          [0, 1, 1, 1, 0],
                          [1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 0],
                          [0, 0, 1, 0, 0]],

                         [[0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0]],

                         [[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]]])

    assert_array_equal(test, expected.astype(bool))
    assert_array_equal(test_anisotropic, expected.astype(bool))


def test_ellipsoid_levelset():
    test = ellipsoid(2, 2, 2, levelset=True)[1:-1, 1:-1, 1:-1]
    test_anisotropic = ellipsoid(2, 2, 4, spacing=(1., 1., 2.),
                                 levelset=True)
    test_anisotropic = test_anisotropic[1:-1, 1:-1, 1:-1]

    expected = np.array([[[ 2.  ,  1.25,  1.  ,  1.25,  2.  ],
                          [ 1.25,  0.5 ,  0.25,  0.5 ,  1.25],
                          [ 1.  ,  0.25,  0.  ,  0.25,  1.  ],
                          [ 1.25,  0.5 ,  0.25,  0.5 ,  1.25],
                          [ 2.  ,  1.25,  1.  ,  1.25,  2.  ]],

                         [[ 1.25,  0.5 ,  0.25,  0.5 ,  1.25],
                          [ 0.5 , -0.25, -0.5 , -0.25,  0.5 ],
                          [ 0.25, -0.5 , -0.75, -0.5 ,  0.25],
                          [ 0.5 , -0.25, -0.5 , -0.25,  0.5 ],
                          [ 1.25,  0.5 ,  0.25,  0.5 ,  1.25]],

                         [[ 1.  ,  0.25,  0.  ,  0.25,  1.  ],
                          [ 0.25, -0.5 , -0.75, -0.5 ,  0.25],
                          [ 0.  , -0.75, -1.  , -0.75,  0.  ],
                          [ 0.25, -0.5 , -0.75, -0.5 ,  0.25],
                          [ 1.  ,  0.25,  0.  ,  0.25,  1.  ]],

                         [[ 1.25,  0.5 ,  0.25,  0.5 ,  1.25],
                          [ 0.5 , -0.25, -0.5 , -0.25,  0.5 ],
                          [ 0.25, -0.5 , -0.75, -0.5 ,  0.25],
                          [ 0.5 , -0.25, -0.5 , -0.25,  0.5 ],
                          [ 1.25,  0.5 ,  0.25,  0.5 ,  1.25]],

                         [[ 2.  ,  1.25,  1.  ,  1.25,  2.  ],
                          [ 1.25,  0.5 ,  0.25,  0.5 ,  1.25],
                          [ 1.  ,  0.25,  0.  ,  0.25,  1.  ],
                          [ 1.25,  0.5 ,  0.25,  0.5 ,  1.25],
                          [ 2.  ,  1.25,  1.  ,  1.25,  2.  ]]])

    assert_allclose(test, expected)
    assert_allclose(test_anisotropic, expected)


def test_ellipsoid_stats():
    # Test comparison values generated by Wolfram Alpha
    vol, surf = ellipsoid_stats(6, 10, 16)
    assert_allclose(1280 * np.pi, vol, atol=1e-4)
    assert_allclose(1383.28, surf, atol=1e-2)

    # Test when a <= b <= c does not hold
    vol, surf = ellipsoid_stats(16, 6, 10)
    assert_allclose(1280 * np.pi, vol, atol=1e-4)
    assert_allclose(1383.28, surf, atol=1e-2)

    # Larger test to ensure reliability over broad range
    vol, surf = ellipsoid_stats(17, 27, 169)
    assert_allclose(103428 * np.pi, vol, atol=1e-4)
    assert_allclose(37426.3, surf, atol=1e-1)


if __name__ == "__main__":
    np.testing.run_module_suite()
