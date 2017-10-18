from __future__ import division

import numpy as np

from ..draw import rectangle
from ..exposure import rescale_intensity
from .._shared.utils import check_random_state


def haar_like_feature_visualize(coord, height, width,
                                max_n_features=None, random_state=None):
    """Helper to visualize Haar-like features.

    Parameters
    ----------
    coord : list of tuple coord, shape (n_features, 2, n_rectangles)
        Output of the ``haar_like_feature_coord`` function.

    height : int
        Height of the detection window.

    width : int
        Width of the detection window.

    max_n_features : int, default=None
        The maximum number of features to be returned.
        By default, all features are returned.

    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. The random state is used when generating a set of
        feature smaller than the total number of available features.

    Returns
    -------
    feature_set : (max_n_features, height, width), ndarray
        A set of images plotting the Haar-like features created.

    """
    random_state = check_random_state(random_state)
    if max_n_features is None:
        feature_indices = range(len(coord[0][0]))
    else:
        feature_indices = random_state.choice(
            range(len(coord[0][0])),
            size=max_n_features, replace=False)

    feature_set = [np.zeros((height, width))
                   for _ in range(len(feature_indices))]

    for set_idx, feature_idx in enumerate(feature_indices):
        for idx_rect, rect in enumerate(coord):
            coord_start, coord_end = rect
            rect_coord = rectangle(coord_start[feature_idx],
                                   coord_end[feature_idx])
            feature_set[set_idx][rect_coord] = (idx_rect + 1) % 2

    return feature_set
