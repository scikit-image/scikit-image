import numpy as np


def integral_image(x):
    """Integral image / summed area table.

    The integral image contains the sum of all elements above and to the
    left of it, i.e.:

    .. math::

       S[m, n] = \sum_{i \leq m} \sum_{j \leq n} X[i, j]

    Parameters
    ----------
    x : ndarray
        Input image.

    Returns
    -------
    S : ndarray
        Integral image / summed area table.

    References
    ----------
    .. [1] F.C. Crow, "Summed-area tables for texture mapping,"
           ACM SIGGRAPH Computer Graphics, vol. 18, 1984, pp. 207-212.

    """
    return x.cumsum(1).cumsum(0)


def integrate(ii, r0, c0, r1, c1):
    """Use an integral image to integrate over a given window.

    Parameters
    ----------
    ii : ndarray
        Integral image.
    r0, c0 : int or ndarray
        Top-left corner(s) of block to be summed.
    r1, c1 : int or ndarray
        Bottom-right corner(s) of block to be summed.

    Returns
    -------
    S : scalar or ndarray
        Integral (sum) over the given window(s).

    """
    if np.isscalar(r0):
        r0, c0, r1, c1 = [np.asarray([x]) for x in (r0, c0, r1, c1)]

    S = np.zeros(r0.shape, ii.dtype)

    S += ii[r1, c1]

    good = (r0 >= 1) & (c0 >= 1)
    S[good] += ii[r0[good] - 1, c0[good] - 1]

    good = r0 >= 1
    S[good] -= ii[r0[good] - 1, c1[good]]

    good = c0 >= 1
    S[good] -= ii[r1[good], c0[good] - 1]

    if S.size == 1:
        return np.asscalar(S)

    return S
