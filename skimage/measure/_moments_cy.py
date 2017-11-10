import numpy as np
import numba


@numba.jit(nogil=True)
def moments_central(image, cr, cc, order):
    mu = np.zeros((order + 1, order + 1), dtype=np.double)
    for r in range(image.shape[0]):
        dr = r - cr
        for c in range(image.shape[1]):
            dc = c - cc
            val = image[r, c]
            dcp = 1
            for p in range(order + 1):
                drq = 1
                for q in range(order + 1):
                    mu[p, q] += val * drq * dcp
                    drq *= dr
                dcp *= dc
    return mu


@numba.jit(nogil=True)
def moments_normalized(mu, order=3):
    nu = np.zeros((order + 1, order + 1), dtype=np.double)
    for p in range(order + 1):
        for q in range(order + 1):
            if p + q >= 2:
                nu[p, q] = mu[p, q] / mu[0, 0] ** ((p + q) / 2 + 1)
            else:
                nu[p, q] = np.nan
    return nu


@numba.jit(nogil=True)
def moments_hu(nu):
    hu = np.zeros((7, ), dtype=np.double)
    t0 = nu[3, 0] + nu[1, 2]
    t1 = nu[2, 1] + nu[0, 3]
    q0 = t0 * t0
    q1 = t1 * t1
    n4 = 4 * nu[1, 1]
    s = nu[2, 0] + nu[0, 2]
    d = nu[2, 0] - nu[0, 2]
    hu[0] = s
    hu[1] = d * d + n4 * nu[1, 1]
    hu[3] = q0 + q1
    hu[5] = d * (q0 - q1) + n4 * t0 * t1
    t0 *= q0 - 3 * q1
    t1 *= 3 * q0 - q1
    q0 = nu[3, 0]- 3 * nu[1, 2]
    q1 = 3 * nu[2, 1] - nu[0, 3]
    hu[2] = q0 * q0 + q1 * q1
    hu[4] = q0 * t0 + q1 * t1
    hu[6] = q1 * t0 - q0 * t1
    return hu
