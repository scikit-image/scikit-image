#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.stdlib cimport malloc, free, realloc

import numpy as np

from ..transform import integral_image
from .._shared.transform cimport integrate

FEATURE_TYPE = {'type-2-x': 0, 'type-2-y': 1,
                'type-3-x': 2, 'type-3-y': 3,
                'type-4': 4}


cdef Rectangle** _haar_like_feature_coord(unsigned int feature_type,
                                          Py_ssize_t height,
                                          Py_ssize_t width,
                                          Py_ssize_t* n_rectangle,
                                          Py_ssize_t* n_feature) nogil:
    """Private function to compute the coordinates of all Haar-like features.
    """
    # allocate for the worst case scenario
    cdef:
        Py_ssize_t max_feature = height ** 2 * width ** 2
        Rectangle** rect_feat = NULL
        Py_ssize_t cnt_feat = 0
        Py_ssize_t local_n_rectangle
        Py_ssize_t x, y, dx, dy

    if feature_type == 0 or feature_type == 1:
        local_n_rectangle = 2
    elif feature_type == 2 or feature_type == 3:
        local_n_rectangle = 3
    else:
        local_n_rectangle = 4
    n_rectangle[0] = local_n_rectangle

    rect_feat = <Rectangle**> malloc(local_n_rectangle * sizeof(Rectangle*))
    for rect_idx in range(local_n_rectangle):
        rect_feat[rect_idx] = <Rectangle*> malloc(max_feature *
                                                  sizeof(Rectangle))

    for y in range(height):
        for x in range(width):
            for dy in range(1, height):
                for dx in range(1, width):
                    # type -> 2 rectangles split along x axis
                    if (feature_type == 0 and
                            (y + dy <= height and x + 2 * dx <= width)):
                        set_rectangle_feature(&rect_feat[0][cnt_feat],
                                              y, x,
                                              y + dy - 1, x + dx - 1)
                        set_rectangle_feature(&rect_feat[1][cnt_feat],
                                              y, x + dx,
                                              y + dy - 1, x + 2 * dx - 1)
                        cnt_feat += 1
                    # type -> 2 rectangles split along y axis
                    elif (feature_type == 1 and
                          (y + 2 * dy <= height and x + dx <= width)):
                        set_rectangle_feature(&rect_feat[0][cnt_feat],
                                              y, x,
                                              y + dy - 1, x + dx - 1)
                        set_rectangle_feature(&rect_feat[1][cnt_feat],
                                              y + dy, x,
                                              y + 2 * dy - 1, x + dx - 1)
                        cnt_feat += 1
                    # type -> 3 rectangles split along x axis
                    elif (feature_type == 2 and
                          (y + dy <= height and x + 3 * dx <= width)):
                        set_rectangle_feature(&rect_feat[0][cnt_feat],
                                              y, x,
                                              y + dy - 1, x + dx - 1)
                        set_rectangle_feature(&rect_feat[1][cnt_feat],
                                              y, x + dx,
                                              y + dy - 1, x + 2 * dx - 1)
                        set_rectangle_feature(&rect_feat[2][cnt_feat],
                                              y, x + 2 * dx,
                                              y + dy - 1, x + 3 * dx - 1)
                        cnt_feat += 1
                    # type -> 3 rectangles split along y axis
                    elif (feature_type == 3 and
                          (y + 3 * dy <= height and x + dx <= width)):
                        set_rectangle_feature(&rect_feat[0][cnt_feat],
                                              y, x,
                                              y + dy - 1, x + dx - 1)
                        set_rectangle_feature(&rect_feat[1][cnt_feat],
                                              y + dy, x,
                                              y + 2 * dy - 1, x + dx - 1)
                        set_rectangle_feature(&rect_feat[2][cnt_feat],
                                              y + 2 * dy, x,
                                              y + 3 * dy - 1, x + dx - 1)
                        cnt_feat += 1
                    # type -> 4 rectangles split along x and y axis
                    elif (feature_type == 4 and
                          (y + 2 * dy <= height and x + 2 * dx <= width)):
                        set_rectangle_feature(&rect_feat[0][cnt_feat],
                                              y, x,
                                              y + dy - 1, x + dx - 1)
                        set_rectangle_feature(&rect_feat[1][cnt_feat],
                                              y, x + dx,
                                              y + dy - 1, x + 2 * dx - 1)
                        set_rectangle_feature(&rect_feat[3][cnt_feat],
                                              y + dy, x,
                                              y + 2 * dy - 1, x + dx - 1)
                        set_rectangle_feature(&rect_feat[2][cnt_feat],
                                              y + dy, x + dx,
                                              y + 2 * dy - 1, x + 2 * dx - 1)
                        cnt_feat += 1

    for rect_idx in range(local_n_rectangle):
        rect_feat[rect_idx] = <Rectangle*> realloc(
            rect_feat[rect_idx], cnt_feat * sizeof(Rectangle))
    n_feature[0] = cnt_feat

    return rect_feat


cpdef haar_like_feature_coord(feature_type, height, width):
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
    >>> from skimage.feature import haar_like_feature_visualize
    >>> coord = haar_like_feature_coord('type-4', 2, 2)
    >>> coord
    [[[(0, 0)], [(0, 0)]],
     [[(0, 1)], [(0, 1)]],
     [[(1, 1)], [(1, 1)]],
     [[(1, 0)], [(1, 0)]]]
    >>> feature = haar_like_feature_visualize(coord, 2, 2, max_n_features=1)
    >>> feature
    [array([[ 1.,  0.],
            [ 0.,  1.]])]

    """
    cdef:
        Rectangle** rect = NULL
        Py_ssize_t n_rectangle, n_feature
        Py_ssize_t i, j
        # cast the height and width to the right type
        Py_ssize_t height_win = <Py_ssize_t> height
        Py_ssize_t width_win = <Py_ssize_t> width

    rect = _haar_like_feature_coord(FEATURE_TYPE[feature_type],
                                    height_win, width_win,
                                    &n_rectangle, &n_feature)

    # allocate the output based on the number of rectangle
    output = [[[], []] for _ in range(n_rectangle)]
    for i in range(n_rectangle):
        for j in range(n_feature):
            output[i][0].append((rect[i][j].top_left.row,
                                 rect[i][j].top_left.col))
            output[i][1].append((rect[i][j].bottom_right.row,
                                 rect[i][j].bottom_right.col))

    return output


cdef integral_floating[:, ::1] _haar_like_feature(
        integral_floating[:, ::1] int_image,
        Rectangle** coord,
        Py_ssize_t n_rectangle, Py_ssize_t n_feature):
    """Private function releasing the GIL to compute the integral for the
    different rectangle."""
    cdef:
        integral_floating[:, ::1] rect_feature = np.empty(
            (n_rectangle, n_feature), dtype=int_image.base.dtype)

        Py_ssize_t idx_rect, idx_feature

    with nogil:
        for idx_rect in range(n_rectangle):
            for idx_feature in range(n_feature):
                rect_feature[idx_rect, idx_feature] = integrate(
                    int_image,
                    coord[idx_rect][idx_feature].top_left.row,
                    coord[idx_rect][idx_feature].top_left.col,
                    coord[idx_rect][idx_feature].bottom_right.row,
                    coord[idx_rect][idx_feature].bottom_right.col)

    return rect_feature


cpdef haar_like_feature(integral_floating[:, ::1] int_image,
                        r, c, width, height, feature_type):
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
    >>> feature = haar_like_feature(img_ii, 'type-4')
    >>> feature
    array([-1, -2, -2, -4, -1, -2, -2, -4, -1, -2, -1, -2, -1, -2, -2, -4, -1,
           -2, -2, -4, -1, -2, -1, -2, -1, -2, -1, -2, -1, -1, -1, -2, -1, -2,
           -1, -1])

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
    cdef:
        Rectangle** coord = NULL
        Py_ssize_t n_rectangle, n_feature
        Py_ssize_t idx_rect, idx_feature
        integral_floating[:, ::1] rect_feature

    if feature_type not in FEATURE_TYPE.keys():
        raise ValueError('The given feature type is unknown. Got {}'
                         ' instead of one of {}.'.format(feature_type,
                                                         FEATURE_TYPE))

    # compute all possible coordinates with a specific type of feature
    coord = _haar_like_feature_coord(FEATURE_TYPE[feature_type],
                                     height,
                                     width,
                                     &n_rectangle, &n_feature)

    rect_feature = _haar_like_feature(int_image[r : r + height,
                                                c : c + width],
                                      coord, n_rectangle, n_feature)

    # deallocate
    for idx_rect in range(n_rectangle):
        free(coord[idx_rect])
    free(coord)

    # convert the memory view to numpy array and convert it to signed array if
    # necessary to avoid overflow during subtraction
    rect_feature_ndarray = np.asarray(rect_feature)
    data_type = rect_feature_ndarray.dtype
    if 'uint' in data_type.name:
        rect_feature_ndarray = rect_feature_ndarray.astype(
            data_type.name.replace('u', ''))

    # the rectangles with odd indices can always be subtracted to the rectangle
    # with even indices
    return (np.sum(rect_feature_ndarray[1::2], axis=0) -
            np.sum(rect_feature_ndarray[::2], axis=0))
