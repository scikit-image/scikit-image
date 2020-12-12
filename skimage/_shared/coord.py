import numpy as np
from scipy.spatial import cKDTree, distance


def _ensure_spacing(coord, spacing, p_norm):
    """Returns a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coord : ndarray
        The coordinates of the considered points.
    spacing : float
        the maximum allowed spacing between the points.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.

    Returns
    -------
    output : ndarray
        A subset of coord where a minimum spacing is guaranteed.

    """

    # Use KDtree to find the peaks that are too close to each other
    tree = cKDTree(coord)

    indices = tree.query_ball_point(coord, r=spacing, p=p_norm)
    rejected_peaks_indices = set()
    for idx, candidates in enumerate(indices):
        if idx not in rejected_peaks_indices:
            # keep current point and the points at exactly spacing from it
            candidates.remove(idx)
            dist = distance.cdist([coord[idx]],
                                  coord[candidates],
                                  distance.minkowski,
                                  p=p_norm).reshape(-1)
            candidates = [c for c, d in zip(candidates, dist)
                          if d < spacing]

            # candidates.remove(keep)
            rejected_peaks_indices.update(candidates)

    # Remove the peaks that are too close to each other
    output = np.delete(coord, tuple(rejected_peaks_indices), axis=0)

    return output


def ensure_spacing(coords, spacing=1, p_norm=np.inf, min_split_size=50):
    """Returns a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coord : array_like
        The coordinates of the considered points.
    spacing : float
        the maximum allowed spacing between the points.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.
    min_split_size : int
        Minimum split size used to process ``coord`` by batch to save
        memory. If None, the memory saving strategy is not applied.

    Returns
    -------
    output : array_like
        A subset of coord where a minimum spacing is guaranteed.

    """

    output = coords
    if len(coords):

        coords = np.atleast_2d(coords)
        if min_split_size is None:
            batch_list = [coords]
        else:
            coord_count = len(coords)
            split_count = int(np.log2(coord_count / min_split_size)) + 1
            split_idx = np.cumsum(
                [coord_count // (2 ** i) for i in range(1, split_count)])
            batch_list = np.array_split(coords, split_idx)

        output = np.zeros((0, coords.shape[1]), dtype=coords.dtype)
        for batch in batch_list:
            output = _ensure_spacing(np.vstack([output, batch]),
                                     spacing, p_norm)

    return output
