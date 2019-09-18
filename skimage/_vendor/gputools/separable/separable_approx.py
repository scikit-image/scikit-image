from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from scipy import linalg
from six.moves import range
from six.moves import zip
from sktensor import dtensor, cp_als

def _separable_series2(h, N=1):
    """ finds separable approximations to the 2d function 2d h

    returns res = (hx, hy)[N]
    s.t. h \approx sum_i outer(res[i,0],res[i,1])
    """
    if min(h.shape)<N:
        raise ValueError("smallest dimension of h is smaller than approximation order! (%s < %s)"%(min(h.shape),N))

    U, S, V = linalg.svd(h)

    hx = [-U[:, n] * np.sqrt(S[n]) for n in range(N)]
    hy = [-V[n, :] * np.sqrt(S[n]) for n in range(N)]
    return np.array(list(zip(hx, hy)))


def _separable_approx2(h, N=1):
    """ returns the N first approximations to the 2d function h
    whose sum should be h
    """
    return np.cumsum([np.outer(fy, fx) for fy, fx in _separable_series2(h, N)], 0)


# def __separable_series3(h, N=1, verbose=False):
#     """ finds separable approximations to the 3d kernel h
#     returns res = (hx,hy,hz)[N]
#     s.t. h \approx sum_i outer(res[i,0],res[i,1],res[i,2])
#     FIXME: This is just a naive and slow first try!
#     """
#     hx, hy, hz = [], [], []
#     u = h.copy()
#     P, fit, itr, exectimes = cp_als(dtensor(u), N)
#     hx, hy, hz = [[(P.lmbda[n]) ** (1. / 3) * np.array(P.U[i])[:, n] for n in range(N)] for i in range(3)]
#     if verbose:
#         print("lambdas= %s \nfit = %s \niterations= %s " % (P.lmbda, fit, itr))
#     return np.array(zip(hx, hy, hz))


def _splitrank3(h, verbose=False):
    # this fixes a bug in scikit-tensor
    if np.sum(np.abs(h))<1.e-30:
        return tuple(np.zeros(s) for s in h.shape)+(np.zeros_like(h),)

    P, fit, itr = cp_als(dtensor(h.copy()), 1)[:3]  # ensure backwards compat
    hx, hy, hz = [(P.lmbda[0]) ** (1. / 3) * np.array(P.U[i])[:, 0] for i in range(3)]
    if verbose:
        print("lambdas= %s \nfit = %s \niterations= %s " % (P.lmbda, fit, itr))
    return hx, hy, hz, P.toarray()


def _separable_series3(h, N=1, verbose=False):
    """ finds separable approximations to the 3d kernel h
    returns res = (hx,hy,hz)[N]
    s.t. h \approx sum_i einsum("i,j,k",res[i,0],res[i,1],res[i,2])

    FIXME: This is just a naive and slow first try!
    """

    hx, hy, hz = [], [], []
    res = h.copy()
    for i in range(N):
        _hx, _hy, _hz, P = _splitrank3(res, verbose=verbose)
        res -= P
        hx.append(_hx)
        hy.append(_hy)
        hz.append(_hz)
    return np.array(list(zip(hx, hy, hz)))


def _separable_approx3(h, N=1):
    """ returns the N first approximations to the 3d function h
    """
    return np.cumsum([np.einsum("i,j,k", fz, fy, fx) for fz, fy, fx in _separable_series3(h, N)], 0)


##################################################################################################



def separable_series(h, N=1):
    """
    finds the first N rank 1 tensors such that their sum approximates
    the tensor h (2d or 3d) best

    returns (e.g. for 3d case) res = (hx,hy,hz)[i]

    s.t.

    h \approx sum_i einsum("i,j,k",res[i,0],res[i,1],res[i,2])

    Parameters
    ----------
    h: ndarray
        input array (2 or 2 dimensional)
    N: int
        order of approximation

    Returns
    -------
        res, the series of tensors
         res[i] = (hx,hy,hz)[i]

    """
    if h.ndim == 2:
        return _separable_series2(h, N)
    elif h.ndim == 3:
        return _separable_series3(h, N)
    else:
        raise ValueError("unsupported array dimension: %s (only 2d or 3d) " % h.ndim)


def separable_approx(h, N=1):
    """
    finds the k-th rank approximation to h, where k = 1..N

    similar to separable_series

    Parameters
    ----------
    h: ndarray
        input array (2 or 2 dimensional)
    N: int
        order of approximation

    Returns
    -------
        all N apprxoimations res[i], the i-th approximation

    """
    if h.ndim == 2:
        return _separable_approx2(h, N)
    elif h.ndim == 3:
        return _separable_approx3(h, N)
    else:
        raise ValueError("unsupported array dimension: %s (only 2d or 3d) " % h.ndim)
