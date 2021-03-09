import numpy as np
from skimage._shared.testing import assert_array_equal, assert_allclose

from skimage.draw import (ellipsoid, ellipsoid_coords, ellipsoid_stats,
                          rectangle)
from skimage._shared import testing


def test_ellipsoid_sign_parameters1():
    with testing.raises(ValueError):
        ellipsoid(-1, 2, 2)


def test_ellipsoid_sign_parameters2():
    with testing.raises(ValueError):
        ellipsoid(0, 2, 2)


def test_ellipsoid_sign_parameters3():
    with testing.raises(ValueError):
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


def test_ellipsoid_coords():
    test = np.zeros((5, 5, 5))
    dd, rr, cc = ellipsoid_coords(2, 2, 2, 2.2, 2.2, 2.2)
    test[dd, rr, cc] = 1
    test_anisotropic = np.zeros((5, 5, 5))
    dd, rr, cc = ellipsoid_coords(2, 2, 4, 2.2, 2.2, 4.4, spacing=(1., 1., 2.))
    test_anisotropic[dd, rr, cc] = 1
    test_rotate = np.zeros((5, 5, 5))
    dd, rr, cc = ellipsoid_coords(2, 2, 2, 2.2, 2.2, 2.2,
                                  rot_z=np.random.uniform(np.pi),
                                  rot_y=np.random.uniform(np.pi),
                                  rot_x=np.random.uniform(np.pi))
    test_rotate[dd, rr, cc] = 1
    test_rotate_anisotropic = np.zeros((5, 5, 5))
    dd, rr, cc = ellipsoid_coords(4, 2, 2, 2.2, 2.2, 4.4, rot_y=np.pi/2,
                                  spacing=(2., 1., 1.))
    test_rotate_anisotropic[dd, rr, cc] = 1

    expected = np.array([[[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 1., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]],

                         [[0., 0., 0., 0., 0.],
                          [0., 1., 1., 1., 0.],
                          [0., 1., 1., 1., 0.],
                          [0., 1., 1., 1., 0.],
                          [0., 0., 0., 0., 0.]],

                         [[0., 0., 1., 0., 0.],
                          [0., 1., 1., 1., 0.],
                          [1., 1., 1., 1., 1.],
                          [0., 1., 1., 1., 0.],
                          [0., 0., 1., 0., 0.]],

                         [[0., 0., 0., 0., 0.],
                          [0., 1., 1., 1., 0.],
                          [0., 1., 1., 1., 0.],
                          [0., 1., 1., 1., 0.],
                          [0., 0., 0., 0., 0.]],

                         [[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 1., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]]])

    assert_array_equal(test, expected)
    assert_array_equal(test_anisotropic, expected)
    assert_array_equal(test_rotate, expected)
    assert_array_equal(test_rotate_anisotropic, expected)


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


def test_rect_3d_extent():
    expected = np.array([[[0, 0, 1, 1, 1],
                          [0, 0, 1, 1, 1],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]],
                         [[0, 0, 1, 1, 1],
                          [0, 0, 1, 1, 1],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]],
                         [[0, 0, 1, 1, 1],
                          [0, 0, 1, 1, 1],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]],
                         [[0, 0, 1, 1, 1],
                          [0, 0, 1, 1, 1],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]]], dtype=np.uint8)
    img = np.zeros((4, 5, 5), dtype=np.uint8)
    start = (0, 0, 2)
    extent = (5, 2, 3)
    pp, rr, cc = rectangle(start, extent=extent, shape=img.shape)
    img[pp, rr, cc] = 1
    assert_array_equal(img, expected)


def test_rect_3d_end():
    expected = np.array([[[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]],
                         [[0, 0, 1, 1, 0],
                          [0, 0, 1, 1, 0],
                          [0, 0, 1, 1, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]],
                         [[0, 0, 1, 1, 0],
                          [0, 0, 1, 1, 0],
                          [0, 0, 1, 1, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]],
                         [[0, 0, 1, 1, 0],
                          [0, 0, 1, 1, 0],
                          [0, 0, 1, 1, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]]], dtype=np.uint8)
    img = np.zeros((4, 5, 5), dtype=np.uint8)
    start = (1, 0, 2)
    end = (3, 2, 3)
    pp, rr, cc = rectangle(start, end=end, shape=img.shape)
    img[pp, rr, cc] = 1
    assert_array_equal(img, expected)
