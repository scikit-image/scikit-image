import numpy as np


def regular_grid(ar_shape, n_points):
    """Find `n_points` regularly spaced along `ar_shape`.
    
    The returned points (as slices) should be as close to cubically-spaced as
    possible.

    Parameters
    ----------
    ar_shape : array-like of ints
        The shape of the space embedding the grid. `len(ar_shape)` is the
        number of dimensions.
    n_points : int
        The (approximate) number of points to embed in the space.

    Returns
    -------
    slices : list of slice objects
        A slice along each dimension of `ar_shape`, such that the intersection
        of all the slices give the coordinates of regularly spaced points.
    """
    ar_shape = np.asanyarray(ar_shape)
    ndim = len(ar_shape)
    unsort_dim_idxs = np.argsort(np.argsort(ar_shape))
    sorted_dims = np.sort(ar_shape)
    space_size = float(np.prod(ar_shape))
    if space_size <= n_points:
        return [slice(None)] * ndim
    stepsizes = (space_size / n_points) ** (1.0 / ndim) * np.ones(ndim)
    if (sorted_dims < stepsizes).any():
        for dim in range(ndim):
            stepsizes[dim] = sorted_dims[dim]
            space_size = float(np.prod(sorted_dims[dim+1:]))
            stepsizes[dim+1:] = ((space_size / n_points) **
                                            (1.0 / (ndim - dim - 1)))
            if (sorted_dims >= stepsizes).all():
                break
    starts = stepsizes // 2
    stepsizes = np.round(stepsizes)
    slices = [slice(start, None, step) for 
                                        start, step in zip(starts, stepsizes)]
    slices = [slices[i] for i in unsort_dim_idxs]
    return slices
