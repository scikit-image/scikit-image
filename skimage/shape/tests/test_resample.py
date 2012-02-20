from nose.tools import raises
from numpy.testing import assert_almost_equal as almost

import numpy as np

from skimage.shape import resample


@raises(AssertionError)
def test_wrong_ndim():
    arr = np.random.randn(2, 3, 4, 5)
    resample(arr, (1, 2, 3, 4))


@raises(AssertionError)
def test_wrong_intp2d_type():
    arr = np.random.randn(2, 3, 4)
    resample(arr, (1, 2, 3), intp2d=2.3)


@raises(AssertionError)
def test_wrong_order():
    arr = np.random.randn(2, 3, 4)
    resample(arr, (1, 2, 3), order=6)


@raises(AssertionError)
def test_wrong_third_dimension():
    arr = np.random.randn(2, 3, 4)
    resample(arr, (1, 2, 3), intp2d=True)


def test_intp_of_constant_arr():
    h_in, w_in, d_in = 10, 20, 5
    h_out, w_out, d_out = 5., 25., 5.
    x_in, y_in, z_in = np.mgrid[:h_in, :w_in, :d_in]
    x_out, y_out, z_out = np.mgrid[:h_out, :w_out, :d_out]
    x_out = 1. * x_out / x_out.max() * (h_in - 1.)
    y_out = 1. * y_out / y_out.max() * (w_in - 1.)
    z_out = 1. * z_out / z_out.max() * (d_in - 1.)

    # -- arbitrary constant function
    def f0(x, y):
        a = 1.234
        return a * np.ones(x.shape)

    ref_out = f0(x_out, y_out)
    out0 = resample(f0(x_in, y_in), (h_out, w_out, d_out), order=0)
    almost(ref_out, out0)
    out1 = resample(f0(x_in, y_in), (h_out, w_out, d_out), order=1)
    almost(ref_out, out1)
    out2 = resample(f0(x_in, y_in), (h_out, w_out, d_out), order=2)
    almost(ref_out, out2)
    out3 = resample(f0(x_in, y_in), (h_out, w_out, d_out), order=3)
    almost(ref_out, out3)
    out4 = resample(f0(x_in, y_in), (h_out, w_out, d_out), order=4)
    almost(ref_out, out4)
    out5 = resample(f0(x_in, y_in), (h_out, w_out, d_out), order=5)
    almost(ref_out, out5)


def test_intp_of_linear_arr():
    h_in, w_in, d_in = 10, 20, 5
    h_out, w_out, d_out = 5., 25., 5.
    x_in, y_in, z_in = np.mgrid[:h_in, :w_in, :d_in]
    x_out, y_out, z_out = np.mgrid[:h_out, :w_out, :d_out]
    x_out = 1. * x_out / x_out.max() * (h_in - 1.)
    y_out = 1. * y_out / y_out.max() * (w_in - 1.)
    z_out = 1. * z_out / z_out.max() * (d_in - 1.)

    # -- arbitray linear function
    def f1(x, y):
        a = 1.234
        b = 0.987
        return a * x + b * y

    ref_out = f1(x_out, y_out)
    out1 = resample(f1(x_in, y_in), (h_out, w_out, d_out), order=1)
    almost(ref_out, out1)
