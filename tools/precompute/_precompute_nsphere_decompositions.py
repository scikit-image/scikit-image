"""Utility script that was used to precompute disk and sphere footprints in
terms of a series of small 3x3 (or 3x3x3) footprints.

This is a crude, brute force implementation that checks many combinations and
retains the one that has the minimum error between the composite footprint and
the desired one.

The generated footprints were stored as
    skimage/morphology/ball_decompositions.npy
    skimage/morphology/disk_decompositions.npy

Validation in `test_nsphere_series_approximation` in:
    skimage/morphology/tests/test_footprints.py
"""

import numpy as np

from skimage.morphology import ball, disk
from skimage.morphology.footprints import (
    _t_shaped_element_series,
    footprint_from_sequence,
)


def precompute_decompositions(
    ndims=[2, 3], radius_max_per_ndim={2: 200, 3: 100}, strict_radius=False
):
    assert all(d in radius_max_per_ndim for d in ndims)

    best_vals = {}
    for ndim in ndims:
        dtype = np.uint8
        # shape (3,) * ndim hyperoctahedron footprint
        # i.e. d = diamond(radius=1) in 2D
        #      d = octahedron(radius=1) in 3D
        d = np.zeros((3,) * ndim, dtype=dtype)
        sl = [
            slice(1, 2),
        ] * ndim
        for ax in range(ndim):
            sl[ax] = slice(None)
            d[tuple(sl)] = 1
            sl[ax] = slice(1, 2)

        # shape (3,) * ndim hypercube footprint
        sq3 = np.ones((3,) * ndim, dtype=dtype)

        # shape (3,) * ndim "T-shaped" footprints
        all_t = _t_shaped_element_series(ndim=ndim, dtype=dtype)

        radius_max = radius_max_per_ndim[ndim]
        for radius in range(2, radius_max + 1):
            if ndim == 2:
                desired = disk(radius, decomposition=None, strict_radius=strict_radius)
            elif ndim == 3:
                desired = ball(radius, decomposition=None, strict_radius=strict_radius)
            else:
                raise ValueError(f"ndim={ndim} not currently supported")

            all_actual = []
            min_err = np.inf
            for n_t in range(radius // len(all_t) + 1):
                if (n_t * len(all_t)) > radius:
                    n_t -= 1
                len_t = n_t * len(all_t)
                d_range = range(radius - len_t, -1, -1)
                err_prev = np.inf
                for n_diamond in d_range:
                    r_rem = radius - len_t - n_diamond
                    n_square = r_rem
                    sequence = []
                    if n_t > 0:
                        sequence += [(t, n_t) for t in all_t]
                    if n_diamond > 0:
                        sequence += [(d, n_diamond)]
                    if n_square > 0:
                        sequence += [(sq3, n_square)]
                    sequence = tuple(sequence)
                    actual = footprint_from_sequence(sequence).astype(int)

                    all_actual.append(actual)
                    error = np.sum(np.abs(desired - actual))  # + 0.01 * n_square
                    if error > err_prev:
                        print(f"break at n_diamond = {n_diamond}")
                        break
                    err_prev = error
                    if error <= min_err:
                        min_err = error
                        best_vals[(ndim, radius)] = (n_t, n_diamond, n_square)

            sequence = []
            n_t, n_diamond, n_square = best_vals[(ndim, radius)]
            print(
                f'radius = {radius}, sum = {desired.sum()}, '
                f'error={min_err}:\n\tn_t={n_t}, '
                f'n_diamond={n_diamond}, n_square={n_square}\n'
            )
            if n_t > 0:
                sequence += [(t, n_t) for t in all_t]
            if n_diamond > 0:
                sequence += [(d, n_diamond)]
            if n_square > 0:
                sequence += [(sq3, n_square)]
            sequence = tuple(sequence)
            actual = footprint_from_sequence(sequence).astype(int)

        opt_vals = np.zeros((radius_max + 1, 3), dtype=np.uint8)
        best_vals[(ndim, 1)] = (0, 0, 1)
        for i in range(1, radius_max + 1):
            opt_vals[i, :] = best_vals[(ndim, i)]

        if ndim == 3:
            fname = "ball_decompositions.npy"
        elif ndim == 2:
            fname = "disk_decompositions.npy"
        else:
            fname = f"{ndim}sphere_decompositions.npy"
        if strict_radius:
            fname = fname.replace(".npy", "_strict.npy")
        np.save(fname, opt_vals)


if __name__ == "__main__":
    precompute_decompositions(
        ndims=[2, 3], radius_max_per_ndim={2: 250, 3: 100}, strict_radius=False
    )
