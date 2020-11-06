import numpy as np
from scipy.spatial import cKDTree, distance


def ensure_spacing(coord, spacing=1, p_norm=np.inf):
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

    output = coord
    if len(coord):
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
