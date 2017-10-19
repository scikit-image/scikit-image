from __future__ import division

from operator import add

import numpy as np

from ._haar import haar_like_feature_coord_wrapper
from ._haar import haar_like_feature_wrapper
from ..color import gray2rgb
from ..draw import rectangle
from ..exposure import rescale_intensity
from .._shared.utils import check_random_state
from ..util import img_as_float


def haar_like_feature_coord(feature_type, height, width):
    """Compute the coordinates of Haar-like features.

    Parameters
    ----------
    feature_type : str
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
        Coordinates of the rectangles for each
        feature. ``feature_coord[0][0][10]`` corresponds to the top-left corner
        of the first rectangle of the tenth feature while
        ``feature_coord[1][1][10]`` corresponds to the bottom-left corner of
        the second rectangle of the tenth feature. A corner is reprented by a
        tuple (row, col) which can be easily used in the function
        :func:`skimage.draw.rectangle` for instance.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.transform import integral_image
    >>> from skimage.feature import haar_like_feature_coord
    >>> coord = haar_like_feature_coord('type-4', 2, 2)
    >>> coord # doctest: +NORMALIZE_WHITESPACE
    [[[(0, 0)], [(0, 0)]],
     [[(0, 1)], [(0, 1)]],
     [[(1, 1)], [(1, 1)]],
     [[(1, 0)], [(1, 0)]]]

    """
    return haar_like_feature_coord_wrapper(feature_type, height, width)

def haar_like_feature(int_image, r, c, width, height, feature_type):
    """Compute the Haar-like features for a region of interest (ROI) of an
    integral image.

    Haar-like features have been successively used in different computer vision
    applications to detect different targets, objects, etc. It was first
    introduced in [1]_ and has been widely used for real-time face detection
    algorithm proposed in [2]_.

    Parameters
    ----------
    int_image : (M, N) ndarray
        Integral image for which the features need to be computed.

    r : int
        Row-coordinate of top left corner of the detection window.

    c : int
        Column-coordinate of top left corner of the detection window.

    width : int
        Width of the detection window.

    height : int
        Height of the detection window.

    feature_type : str
        The type of feature to consider:

        - 'type-2-x': 2 rectangles varying along the x axis;
        - 'type-2-y': 2 rectangles varying along the y axis;
        - 'type-3-x': 3 rectangles varying along the x axis;
        - 'type-3-y': 3 rectangles varying along the y axis;
        - 'type-4': 4 rectangles varying along x and y axis.

    Returns
    -------
    haar_features : (n_features,) ndarray
        Resulting Haar-like features.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.transform import integral_image
    >>> from skimage.feature import haar_like_feature
    >>> img = np.ones((5, 5), dtype=np.uint8)
    >>> img_ii = integral_image(img)
    >>> feature = haar_like_feature(img_ii, 0, 0, 5, 5, 'type-3-x')
    >>> feature
    array([-1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1,
           -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -1, -2, -3, -1, -2, -3, -1,
           -2, -1, -2, -1, -2, -1, -1, -1])

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Haar-like_feature

    .. [2] Oren, M., Papageorgiou, C., Sinha, P., Osuna, E., & Poggio, T.
           (1997, June). Pedestrian detection using wavelet templates.
           In Computer Vision and Pattern Recognition, 1997. Proceedings.,
           1997 IEEE Computer Society Conference on (pp. 193-199). IEEE.
           http://tinyurl.com/y6ulxfta
           DOI: 10.1109/CVPR.1997.609319

    .. [3] Viola, Paul, and Michael J. Jones. "Robust real-time face
           detection." International journal of computer vision 57.2
           (2004): 137-154.
           http://www.merl.com/publications/docs/TR2004-043.pdf
           DOI: 10.1109/CVPR.2001.990517

    """
    return haar_like_feature_wrapper(int_image, r, c, width, height,
                                     feature_type)

def draw_haar_like_feature(image, r, c, height, width, feature_type,
                           color_positive_block=(1., 0., 0.),
                           color_negative_block=(0., 1., 0.),
                           alpha=0.5, max_n_features=None, random_state=None):
    """Helper to visualize Haar-like features.

    Parameters
    ----------
    image : (M, N) ndarray
        The region of an integral image for which the features need to be
        computed.

    r : int
        Row-coordinate of top left corner of the detection window.

    c : int
        Column-coordinate of top left corner of the detection window.

    width : int
        Width of the detection window.

    height : int
        Height of the detection window.

    feature_type : str
        The type of feature to consider:

        - 'type-2-x': 2 rectangles varying along the x axis;
        - 'type-2-y': 2 rectangles varying along the y axis;
        - 'type-3-x': 3 rectangles varying along the x axis;
        - 'type-3-y': 3 rectangles varying along the y axis;
        - 'type-4': 4 rectangles varying along x and y axis.

    color_positive_rectangle : tuple of 3 floats
        Floats specifying the color for the positive block. Corresponding
        values define (R, G, B) values. Default value is red (1, 0, 0).

    color_negative_block : tuple of 3 floats
        Floats specifying the color for the negative block Corresponding values
        define (R, G, B) values. Default value is blue (0, 1, 0).

    alpha : float
        Value in the range [0, 1] that specifies opacity of visualization. 1 -
        fully transparent, 0 - opaque.

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
    features : (M, N), ndarray
        An image in which the different features will be added.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.transform import integral_image
    >>> from skimage.feature import draw_haar_like_feature
    >>> feature = draw_haar_like_feature(np.zeros((2, 2)),
    ...                                  0, 0, 2, 2,
    ...                                  'type-4',
    ...                                  max_n_features=1)
    >>> feature
    array([[[ 0. ,  0.5,  0. ],
            [ 0.5,  0. ,  0. ]],
    <BLANKLINE>
           [[ 0.5,  0. ,  0. ],
            [ 0. ,  0.5,  0. ]]])

    """

    coord = haar_like_feature_coord(feature_type, height, width)

    color_positive_block = np.asarray(color_positive_block, dtype=np.float64)
    color_negative_block = np.asarray(color_negative_block, dtype=np.float64)

    output = np.copy(image)
    if len(image.shape) < 3:
        output = gray2rgb(image)
    output = img_as_float(output)

    random_state = check_random_state(random_state)
    if max_n_features is None:
        feature_indices = range(len(coord[0][0]))
    else:
        feature_indices = random_state.choice(
            range(len(coord[0][0])),
            size=max_n_features, replace=False)

    for set_idx, feature_idx in enumerate(feature_indices):
        for idx_rect, rect in enumerate(coord):
            coord_start, coord_end = rect
            coord_start = tuple(map(add, coord_start[feature_idx], [r, c]))
            coord_end = tuple(map(add, coord_end[feature_idx], [r, c]))
            rr, cc = rectangle(coord_start, coord_end)

            if ((idx_rect + 1) % 2) == 0:
                new_value = ((1 - alpha) *
                             output[rr, cc] + alpha * color_positive_block)
            else:
                new_value = ((1 - alpha) *
                             output[rr, cc] + alpha * color_negative_block)
            output[rr, cc] = new_value

    return output
