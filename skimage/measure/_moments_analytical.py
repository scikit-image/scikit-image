"""Analytical transformations from raw image moments to central moments.

The expressions for the 2D central moments of order <=2 are often given in
textbooks. Expressions for higher orders and dimensions were generated in SymPy
using ``tools/precompute/moments_sympy.py`` in the GitHub repository.

"""

import itertools
import math

import numpy as np


def _moments_raw_to_central_fast(moments_raw):
    """Analytical formulae for 2D and 3D central moments of order < 4.

    `moments_raw_to_central` will automatically call this function when
    ndim < 4 and order < 4.

    Parameters
    ----------
    moments_raw : ndarray
        The raw moments.

    Returns
    -------
    moments_central : ndarray
        The central moments.
    """
    # --- OPT: Cast once, ensure float64 computation for all ---
    float_dtype = moments_raw.dtype
    m = (
        moments_raw
        if moments_raw.dtype == np.float64
        else moments_raw.astype(np.float64, copy=False)
    )
    ndim = m.ndim
    order = m.shape[0] - 1
    if order >= 4 or ndim not in [2, 3]:
        raise ValueError("This function only supports 2D or 3D moments of order < 4.")

    # --- OPT: Compute minimal needed output shape ---
    moments_central = np.zeros(m.shape, dtype=np.float64)

    if ndim == 2:
        m00 = m[0, 0]
        cx = m[1, 0] / m00
        cy = m[0, 1] / m00
        moments_central[0, 0] = m00
        if order > 1:
            m01 = m[0, 1]
            m10 = m[1, 0]
            moments_central[1, 1] = m[1, 1] - cx * m01
            moments_central[2, 0] = m[2, 0] - cx * m10
            moments_central[0, 2] = m[0, 2] - cy * m01
        if order > 2:
            cx2 = cx * cx
            cy2 = cy * cy
            cxcy = cx * cy
            m01 = m[0, 1]
            m10 = m[1, 0]
            m11 = m[1, 1]
            m20 = m[2, 0]
            m02 = m[0, 2]
            # 3rd order moments
            moments_central[2, 1] = (
                m[2, 1] - 2 * cx * m11 - cy * m20 + cx2 * m01 + cxcy * m10
            )
            moments_central[1, 2] = m[1, 2] - 2 * cy * m11 - cx * m02 + 2 * cxcy * m01
            moments_central[3, 0] = m[3, 0] - 3 * cx * m20 + 2 * cx2 * m10
            moments_central[0, 3] = m[0, 3] - 3 * cy * m02 + 2 * cy2 * m01
    else:
        # 3D case
        m000 = m[0, 0, 0]
        cx = m[1, 0, 0] / m000
        cy = m[0, 1, 0] / m000
        cz = m[0, 0, 1] / m000
        moments_central[0, 0, 0] = m000
        if order > 1:
            m001 = m[0, 0, 1]
            m010 = m[0, 1, 0]
            m100 = m[1, 0, 0]
            moments_central[0, 0, 2] = -cz * m001 + m[0, 0, 2]
            moments_central[0, 1, 1] = -cy * m001 + m[0, 1, 1]
            moments_central[0, 2, 0] = -cy * m010 + m[0, 2, 0]
            moments_central[1, 0, 1] = -cx * m001 + m[1, 0, 1]
            moments_central[1, 1, 0] = -cx * m010 + m[1, 1, 0]
            moments_central[2, 0, 0] = -cx * m100 + m[2, 0, 0]
        if order > 2:
            # cache powers and repeated indices
            cx2 = cx * cx
            cy2 = cy * cy
            cz2 = cz * cz
            m001 = m[0, 0, 1]
            m011 = m[0, 1, 1]
            m010 = m[0, 1, 0]
            m002 = m[0, 0, 2]
            m012 = m[0, 1, 2]
            m021 = m[0, 2, 1]
            m020 = m[0, 2, 0]
            m003 = m[0, 0, 3]
            m003_ = m[0, 0, 3]
            m100 = m[1, 0, 0]
            m101 = m[1, 0, 1]
            m102 = m[1, 0, 2]
            m110 = m[1, 1, 0]
            m111 = m[1, 1, 1]
            m120 = m[1, 2, 0]
            m200 = m[2, 0, 0]
            m201 = m[2, 0, 1]
            m210 = m[2, 1, 0]
            m300 = m[3, 0, 0]
            # 3rd order moments
            moments_central[0, 0, 3] = 2 * cz2 * m001 - 3 * cz * m002 + m003
            moments_central[0, 1, 2] = -cy * m002 + 2 * cz * (cy * m001 - m011) + m012
            moments_central[0, 2, 1] = (
                cy2 * m001 - 2 * cy * m011 + cz * (cy * m010 - m020) + m021
            )
            moments_central[0, 3, 0] = 2 * cy2 * m010 - 3 * cy * m020 + m[0, 3, 0]
            moments_central[1, 0, 2] = -cx * m002 + 2 * cz * (cx * m001 - m101) + m102
            moments_central[1, 1, 1] = (
                -cx * m011 + cy * (cx * m001 - m101) + cz * (cx * m010 - m110) + m111
            )
            moments_central[1, 2, 0] = -cx * m020 - 2 * cy * (-cx * m010 + m110) + m120
            moments_central[2, 0, 1] = (
                cx2 * m001 - 2 * cx * m101 + cz * (cx * m100 - m200) + m201
            )
            moments_central[2, 1, 0] = (
                cx2 * m010 - 2 * cx * m110 + cy * (cx * m100 - m200) + m210
            )
            moments_central[3, 0, 0] = 2 * cx2 * m100 - 3 * cx * m200 + m300

    # --- OPT: Only cast back if needed ---
    if moments_central.dtype != float_dtype:
        return moments_central.astype(float_dtype, copy=False)
    return moments_central


def moments_raw_to_central(moments_raw):
    ndim = moments_raw.ndim
    order = moments_raw.shape[0] - 1
    if ndim in [2, 3] and order < 4:
        return _moments_raw_to_central_fast(moments_raw)

    moments_central = np.zeros_like(moments_raw)
    m = moments_raw
    # centers as computed in centroid above
    centers = tuple(m[tuple(np.eye(ndim, dtype=int))] / m[(0,) * ndim])

    if ndim == 2:
        # This is the general 2D formula from
        # https://en.wikipedia.org/wiki/Image_moment#Central_moments
        for p in range(order + 1):
            for q in range(order + 1):
                if p + q > order:
                    continue
                for i in range(p + 1):
                    term1 = math.comb(p, i)
                    term1 *= (-centers[0]) ** (p - i)
                    for j in range(q + 1):
                        term2 = math.comb(q, j)
                        term2 *= (-centers[1]) ** (q - j)
                        moments_central[p, q] += term1 * term2 * m[i, j]
        return moments_central

    # The nested loops below are an n-dimensional extension of the 2D formula
    # given at https://en.wikipedia.org/wiki/Image_moment#Central_moments

    # iterate over all [0, order] (inclusive) on each axis
    for orders in itertools.product(*((range(order + 1),) * ndim)):
        # `orders` here is the index into the `moments_central` output array
        if sum(orders) > order:
            # skip any moment that is higher than the requested order
            continue
        # loop over terms from `m` contributing to `moments_central[orders]`
        for idxs in itertools.product(*[range(o + 1) for o in orders]):
            val = m[idxs]
            for i_order, c, idx in zip(orders, centers, idxs):
                val *= math.comb(i_order, idx)
                val *= (-c) ** (i_order - idx)
            moments_central[orders] += val

    return moments_central
