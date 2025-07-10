import numpy as np
from scipy.spatial import KDTree
from numpy.typing import ArrayLike


def ensure_spacing(
    coords: ArrayLike,
    spacing: int | float = 1,
    p_norm: float = np.inf,
    min_split_size: int = 50,
    max_out: int | None = None,
    *,
    max_split_size: int = 2000,
    split_size_grow_factor: float = 2.0,
):
    """Returns a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coords : array_like
        The coordinates of the considered points.
    spacing : float
        The minimum allowed spacing between the points.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.
    min_split_size : int
        Minimum split size used to process ``coords`` by batch to save
        memory. If None, the memory saving strategy is not applied.
    max_out : int
        If not None, only the first ``max_out`` candidates are returned.
    max_split_size : int
        Maximum split size used to process ``coords`` by batch to save
        memory. This number was decided by profiling with a large number
        of points. Too small a number results in too much looping in
        Python instead of C, slowing down the process, while too large
        a number results in large memory allocations, slowdowns, and,
        potentially, in the process being killed -- see gh-6010. See
        benchmark results `here
        <https://github.com/scikit-image/scikit-image/pull/6035#discussion_r751518691>`_.
    split_size_grow_factor : float
        Factor by which the batch size grows.
        The first batch will be of size `min_split_size`,
        and subsequent batches will grow by this factor
        until they reach `max_split_size`.

    Returns
    -------
    output : array_like
        A subset of coord where a minimum spacing is guaranteed.

    """
    coords = np.asarray(coords)
    if coords.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {coords.shape}")
    if spacing <= 0:
        raise ValueError("min_distance must be positive")
    if coords.size == 0:
        return coords

    # Calculate largest possible float (within machine precision) smaller than min_distance
    # to keep points at exactly min_distance (due to how KDTree works)
    r_eff = np.nextafter(spacing, 0.0)

    accepted_points: list[np.ndarray] = []
    accepted_count = 0

    for batch in _make_batches(
        coords,
        axis=0,
        min_size=min_split_size,
        max_size=max_split_size,
        grow_factor=split_size_grow_factor,
    ):
        # Filter batch against already accepted points
        if accepted_points:
            # Query first nearest neighbor within min_distance
            dists, _ = KDTree(np.vstack(accepted_points)).query(
                batch,
                k=1,
                p=p_norm,
                distance_upper_bound=r_eff,
                workers=-1,
            )
            candidates = batch[np.isinf(dists)]
        else:
            candidates = batch

        if candidates.size == 0:
            continue

        # Greedy within-batch filtering
        neighbor_lists = KDTree(candidates).query_ball_point(
            candidates,
            r=r_eff,
            p=p_norm,
            workers=-1,
            return_sorted=True,
        )
        batch_rejected: set[int] = set()
        batch_accepted: list[np.ndarray] = []

        for idx, neighbors in enumerate(neighbor_lists):
            if idx in batch_rejected:
                # Already rejected this point
                continue
            # Accept this point
            batch_accepted.append(candidates[idx])
            # Remove self from neighbors
            neighbors = [i for i in neighbors if i != idx]
            # Reject all remaining neighbors
            batch_rejected.update(neighbors)
            # If we have reached the maximum number of points, stop
            accepted_count += 1
            if max_out is not None and accepted_count == max_out:
                break

        accepted_points.extend(batch_accepted)
        if max_out is not None and accepted_count == max_out:
            break

    if not accepted_points:
        return np.empty((0, coords.shape[1]), dtype=coords.dtype)

    output = np.vstack(accepted_points)
    if max_out is not None:
        output = output[:max_out]
    return output


def _make_batches(
    tensor: np.ndarray,
    axis: int = 0,
    min_size: int | None = 50,
    max_size: int = 2000,
    grow_factor: float = 2.0,
) -> list[np.ndarray]:
    """Create batches of a tensor along a specified axis.

    Parameters
    ----------
    tensor
        Input tensor to be split into batches.
    axis
        Index of the axis along which to split the tensor.
    min_size
        Minimum batch size.
        If None, the tensor is returned as a single batch.
    max_size
        Maximum batch size. Batches will not exceed this many elements.
    grow_factor
        Factor by which the batch size grows.

    Returns
    -------
    Tensor batches along the specified axis.
    """
    if min_size is None:
        return [tensor]
    n_elements = tensor.shape[axis]
    if n_elements <= min_size:
        return [tensor]
    split_indices = [min_size]
    split_size = min_size
    while n_elements - split_indices[-1] > max_size:
        split_size = min(int(np.rint(split_size * grow_factor)), max_size)
        split_indices.append(split_indices[-1] + split_size)
    return np.array_split(tensor, indices_or_sections=split_indices, axis=axis)
