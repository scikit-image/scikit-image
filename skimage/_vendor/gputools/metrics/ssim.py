"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from gputools import OCLArray, OCLProgram
from gputools.convolve import uniform_filter
from gputools.utils._abspath import abspath


def compare_ssim_bare(X, Y, data_range=None):
    if not X.dtype == Y.dtype:
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    K1 = 0.01
    K2 = 0.03
    sigma = 1.5

    use_sample_covariance = True

    win_size = 7

    if np.any((np.asarray(X.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent.")

    if data_range is None:
        dmin, dmax = np.amin(X), np.amax(X)
        data_range = dmax - dmin

    ndim = X.ndim

    filter_func = uniform_filter
    filter_args = {'size': win_size}

    # ndimage filters need floating point data
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    NP = win_size ** ndim

    cov_norm = NP / (NP - 1)  # sample covariance

    # compute (weighted) means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(X * X, **filter_args)
    uyy = filter_func(Y * Y, **filter_args)
    uxy = filter_func(X * Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)


    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))



    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    ss = tuple(slice(pad, s - pad) for s in X.shape)
    # compute (weighted) mean of ssim
    mssim = S[ss].mean()

    return mssim

#
# def ssim(x, y, data_range=None):
#     """compute ssim
#     parameters are like the defaults for skimage.compare_ssim
#
#     """
#     if not x.shape == y.shape:
#         raise ValueError('Input images must have the same dimensions.')
#
#     K1 = 0.01
#     K2 = 0.03
#     sigma = 1.5
#     win_size = 7
#
#     if np.any((np.asarray(x.shape) - win_size) < 0):
#         raise ValueError("win_size exceeds image extent.")
#
#     if data_range is None:
#         dmin, dmax = np.amin(x), np.amax(x)
#         data_range = dmax - dmin
#
#     x_g = OCLArray.from_array(x.astype(np.float32, copy=False))
#     y_g = OCLArray.from_array(y.astype(np.float32, copy=False))
#
#     ndim = x.ndim
#     NP = win_size ** ndim
#     cov_norm = 1. * NP / (NP - 1)  # sample covariance
#
#     filter_func = uniform_filter
#     filter_args = {'size': win_size}
#
#     ux = filter_func(x_g, **filter_args)
#     uy = filter_func(y_g, **filter_args)
#
#     # compute (weighted) variances and covariances
#     uxx = filter_func(x_g * x_g, **filter_args)
#     uyy = filter_func(y_g * y_g, **filter_args)
#     uxy = filter_func(x_g * y_g, **filter_args)
#     vx = cov_norm * (uxx - ux * ux)
#     vy = cov_norm * (uyy - uy * uy)
#     vxy = cov_norm * (uxy - ux * uy)
#
#
#     R = 1. * data_range
#     C1 = (K1 * R) ** 2
#     C2 = (K2 * R) ** 2
#
#     A1, A2, B1, B2 = ((2. * ux * uy + C1,
#                        2. * vxy + C2,
#                        ux ** 2 + uy ** 2 + C1,
#                        vx + vy + C2))
#     D = B1 * B2
#     S = (A1 * A2) / D
#
#     # to avoid edge effects will ignore filter radius strip around edges
#     pad = (win_size - 1) // 2
#
#     ss = tuple(slice(pad, s - pad) for s in x.shape)
#     # compute (weighted) mean of ssim
#     mssim = S.get()[ss].mean()
#
#     return mssim


def ssim(x, y, data_range=None, scaled = False, verbose = False):
    """compute ssim
    parameters are like the defaults for skimage.compare_ssim

    """
    if not x.shape == y.shape:
        raise ValueError('Input images must have the same dimensions.')

    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    win_size = 7


    if scaled:
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # center it first for numerical stability...
        my = np.mean(y)
        mx = np.mean(x)
        y = y - my
        sxy = np.mean(x * y)  # mean(y)=0
        sy = np.std(y)
        a, b = sxy / (sy ** 2 + 1.e-30), mx
        if verbose:
            print("scaling in ssim: y2 = %.2g*y+%.2g" % (a, b-my))
        y = a * y + b

        # my = np.mean(y)
        # y = y - my
        # sxy = np.mean(x * y)  # - np.mean(x) * np.mean(y)
        # sy = np.std(y)
        # sx = np.std(x)
        # mx = np.mean(x)
        # a, b = sx / sy, mx
        # print("scaling in ssim: y2 = %.2g*y+%.2g" % (a, b-my))
        # y = a * y + b
        #

    if np.any((np.asarray(x.shape) - win_size) < 0):
        raise ValueError("win_size exceeds image extent.")

    if data_range is None:
        dmin, dmax = np.amin(x), np.amax(x)
        data_range = dmax - dmin+1.e-10

    x_g = OCLArray.from_array(x.astype(np.float32, copy=False))
    y_g = OCLArray.from_array(y.astype(np.float32, copy=False))

    ndim = x.ndim
    NP = win_size ** ndim
    cov_norm = 1. * NP / (NP - 1)  # sample covariance

    filter_func = uniform_filter
    filter_args = {'size': win_size}

    ux = filter_func(x_g, **filter_args)
    uy = filter_func(y_g, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(x_g * x_g, **filter_args)
    uyy = filter_func(y_g * y_g, **filter_args)
    uxy = filter_func(x_g * y_g, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)


    R = 1. * data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    # save some gpu space by minimizing intermediate buffers

    # A1 = 2. * ux * uy+C1
    A1 = np.float32(2.) * ux
    A1 *= uy
    A1 += np.float32(C1)



    # A2 =  2. * vxy + C2
    # overwrite vxy to save space
    A2 = vxy
    A2 *= np.float32(2.)
    A2 += np.float32(C2)


    # B1 =  ux ** 2 + uy ** 2 + C1
    # overwrite ux to save space
    B1 = ux
    B1 *= ux
    uy *= uy
    B1 += uy
    B1 += np.float32(C1)

    # B2 =  vx + vy + C2
    # overwrite vx to save space
    B2 = vx
    B2 += vy
    B2 += np.float32(C2)

    D = B1
    D *= B2
    S = A1
    S *= A2
    S /= D


    # import time
    # time.sleep(2)
    # return 1

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    ss = tuple(slice(pad, s - pad) for s in x.shape)
    # compute (weighted) mean of ssim
    mssim = S.get()[ss].mean()

    return mssim


if __name__ == '__main__':
    dshape = (40,40)
    d0 =np.zeros(dshape,np.float32)

    ss = tuple(slice(s//4,-s//4) for s in dshape)
    d0[ss] = 10.

    d1 = d0+np.random.uniform(0, 1, dshape).astype(np.float32)

    drange = np.amax(d0) - np.amin(d0)


    m1 = compare_ssim_bare(d0,d1, data_range = drange)
    m2 = ssim(d0,d1)

    print(m1)
    print(m2)
    print(np.sum(np.abs(m1-m2)))


