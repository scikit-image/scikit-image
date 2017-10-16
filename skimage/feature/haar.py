from __future__ import division

import numpy as np

from ..draw import rectangle
from ..exposure import rescale_intensity
from ..transform import integral_image, integrate
from .._shared.utils import check_random_state


def haar_like_feature_coord(feature_type, height, width):
    """Compute the coordinates of Haar-like features.

    Parameters
    ----------
    feature_type : string
        The type of feature to consider:

        - 'type-2-x': 2 rectangles varying along the x axis;
        - 'type-2-y': 2 rectangles varying along the y axis;
        - 'type-3-x': 3 rectangles varying along the x axis;
        - 'type-3-y': 3 rectangles varying along the y axis;
        - 'type-4': 4 rectangles varying along x and y axis.

    height : int
        Height of the detection window.

    width : int
        Width of the detection window.

    Returns
    -------
    feature_coord : list of tuple coord, shape (n_rectangles, 2, n_features)
        Coordinates of the rectangles for each feature.

    """
    if feature_type == 'type-2-x':
        height_feature = range(1, height, 1)
        width_feature = range(2, width, 2)
        output = [[[], []], [[], []]]
    elif feature_type == 'type-2-y':
        height_feature = range(2, height, 2)
        width_feature = range(1, width, 1)
        output = [[[], []], [[], []]]
    elif feature_type == 'type-3-x':
        height_feature = range(1, height, 1)
        width_feature = range(3, width, 3)
        output = [[[], []], [[], []], [[], []]]
    elif feature_type == 'type-3-y':
        height_feature = range(3, height, 3)
        width_feature = range(1, width, 1)
        output = [[[], []], [[], []], [[], []]]
    elif feature_type == 'type-4':
        height_feature = range(2, height, 2)
        width_feature = range(2, width, 2)
        output = [[[], []], [[], []], [[], []], [[], []]]
    else:
        known_feature = ('type-2-x', 'type-2-y',
                         'type-3-x', 'type-3-y',
                         'type-4')
        raise ValueError('The type of feature is unknown. Got {} instead of'
                         ' one of {}'.format(feature_type, known_feature))

    for y in range(height):
        for x in range(width):
            for dy in height_feature:
                for dx in width_feature:
                    if x + dx <= width and y + dy <= height:
                        if feature_type == 'type-2-x':
                            output[0][0].append((y, x))
                            output[0][1].append((y + dy - 1, x + dx // 2 - 1))
                            output[1][0].append((y, x + dx // 2))
                            output[1][1].append((y + dy - 1, x + dx - 1))
                        elif feature_type == 'type-2-y':
                            output[0][0].append((y, x))
                            output[0][1].append((y + dy // 2 - 1, x + dx - 1))
                            output[1][0].append((y + dy // 2, x))
                            output[1][1].append((y + dy - 1, x + dx - 1))
                        elif feature_type == 'type-3-x':
                            output[0][0].append((y, x))
                            output[0][1].append((y + dy - 1, x + dx // 3 - 1))
                            output[1][0].append((y, x + dx // 3))
                            output[1][1].append((y + dy - 1,
                                                 x + 2 * dx // 3 - 1))
                            output[2][0].append((y, x + 2 * dx // 3))
                            output[2][1].append((y + dy - 1, x + dx - 1))
                        elif feature_type == 'type-3-y':
                            output[0][0].append((y, x))
                            output[0][1].append((y + dy // 3 - 1, x + dx - 1))
                            output[1][0].append((y + dy // 3, x))
                            output[1][1].append((y + 2 * dy // 3 - 1,
                                                 x + dx - 1))
                            output[2][0].append((y + 2 * dy // 3, x))
                            output[2][1].append((y + dy - 1, x + dx - 1))
                        elif feature_type == 'type-4':
                            output[0][0].append((y, x))
                            output[0][1].append((y + dy // 2 - 1,
                                                 x + dx // 2 - 1))
                            output[1][0].append((y, x + dx // 2))
                            output[1][1].append((y + dy // 2 - 1, x + dx - 1))
                            output[2][0].append((y + dy // 2, x))
                            output[2][1].append((y + dy - 1, x + dx // 2 - 1))
                            output[3][0].append((y + dy // 2, x + dx // 2))
                            output[3][1].append((y + dy - 1, x + dx - 1))
    return output


def haar_like_feature(roi_img, feature_type):
    """Compute the Haar-like features.

    Parameters
    ----------
    roi : ndarray
        The region of an image for which the features need to be computed.

    feature_type : string
        The type of feature to consider:

        - 'type-2-x': 2 rectangles varying along the x axis;
        - 'type-2-y': 2 rectangles varying along the y axis;
        - 'type-3-x': 3 rectangles varying along the x axis;
        - 'type-3-y': 3 rectangles varying along the y axis;
        - 'type-4': 4 rectangles varying along x and y axis.

    Returns
    -------
    haar_features : ndarray, shape (n_features,)
        Resulting Haar-like features

    """
    roi_img_integral = integral_image(roi_img)

    coord = haar_like_feature_coord(feature_type,
                                    height=roi_img_integral.shape[0],
                                    width=roi_img_integral.shape[1])

    rect_feat = [integrate(roi_img_integral, rect_coord[0], rect_coord[1])
                 for rect_coord in coord]

    # the rectangles with odd indices can always be subtracted to the rectangle
    # with even indices
    return sum([rect if not rect_idx % 2 else -rect
                for rect_idx, rect in enumerate(rect_feat)])


def haar_like_feature_visualize(haar_like_feature, height, width,
                                max_n_features=10, random_state=None):
    """Helper to visualize Haar-like features.

    Parameters
    ----------
    haar_like_feature : list of tuple coord, shape (n_features, n_rectangles)
        Output of the ``haar_like_feature_coord``.

    height : int
        Height of the detection window.

    width : int
        Width of the detection window.

    max_n_features : int
        The maximum number of features to be returned.

    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. The random state is used when generating a set of
        feature smaller than the total number of available features.

    Returns
    -------
    feature_set : list of ndarray, shape (max_n_features, height, width)
        A set of images plotting the Haar-like features created.

    """
    random_state = check_random_state(random_state)
    features_indices = random_state.choice(range(len(haar_like_feature[0][0])),
                                           size=max_n_features, replace=False)

    feature_set = [np.zeros((height, width))
                   for _ in range(len(features_indices))]

    for set_idx, feature_idx in enumerate(features_indices):
        for idx_rect, rect in enumerate(haar_like_feature):
            coord_start, coord_end = rect
            rect_coord = rectangle(coord_start[feature_idx],
                                   coord_end[feature_idx])
            feature_set[set_idx][rect_coord] = idx_rect + 1
        feature_set[set_idx] = rescale_intensity(feature_set[set_idx])

    return feature_set
