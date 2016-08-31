import numpy as np


def xy_to_rc(points=None, image=None):
    """Convert points' coordinates and/or image from `x, y` to `r, c` convention.

        o--->....       o--->....
        |   x   .       |   c   .
        |       .  ==>  |       .
        v y     .       v r     .
        .........       .........

    Parameters
    ----------
    points : (K, 2) ndarray, optional
        Array of points' `x, y` coordinates.
    image : (M, N[, C]) ndarray, optional
        Image stored in `x, y` axes convention.

    Returns
    -------
    points_out : (K, 2) ndarray
        Array of `points` `r, c` coordinates.
    image_out : (M, N[, C]) ndarray
        Same data as `image`, but in `r, c` convention.
    """
    if points is not None:
        points_out = np.fliplr(points)
    else:
        points_out = None

    if image is not None:
        image_out = np.swapaxes(image, 0, 1)
    else:
        image_out = None

    return points_out, image_out


def rc_to_xy(points=None, image=None):
    """Convert points' coordinates and/or image from `r, c` to `x, y` convention.

        o--->....       o--->....
        |   c   .       |   x   .
        |       .  ==>  |       .
        v r     .       v y     .
        .........       .........

    Parameters
    ----------
image : (M, N[, C]) ndarray, optional
        Image stored in `r, c` axes convention.
    points : (K, 2) ndarray, optional
        Array of points' `r, c` coordinates.
    copy : bool, optional
        If True, returns copy(s) of array(s) instead of view(s).

    Returns
    -------
    image_out : (M, N[, C]) ndarray
        Same as `image`, but in `x, y` convention.
    points_out : (K, 2) ndarray
        Array of `points` `x, y` coordinates.
    """
    if points is not None:
        points_out = np.fliplr(points)
    else:
        points_out = None

    if image is not None:
        image_out = np.swapaxes(image, 0, 1)
    else:
        image_out = None

    return points_out, image_out


def cart_to_rc(points=None, deltao=None, image=None):
    """Convert image and/or points' coordinates from `x_c, y_c` to `r, c` convention.

        .........       o--->....
        ^ y_c   .       |   c   .
        |       .  ==>  |       .
        |   x_c .       v r     .
        o--->....       .........

    Parameters
    ----------
    points : (K, 2) ndarray, optional
        Array of points' `x_c, y_c` Cartesian coordinates.
    deltao : int, optional
        Distance between `y_c` and `r` vectors' origins.
        Optional for `points` convertion if `image` is specified.
    image : (M, N[, C]) ndarray, optional
        Image stored in `r, c` axes convention.

    Returns
    -------
    points_out : (K, 2) ndarray
        Array of `points` `r, c` coordinates.
    image_out : (M, N[, C]) ndarray
        Same data as `image`, but in `r, c` convention.
    """
    if points is not None:
        if deltao is None:
            if image is None:
                msg = ('When converting coordinates, either `deltao` or `image` '
                       'have to be specified.')
                raise ValueError(msg)
            else:
                deltao = image.shape[1] - 1
        r = deltao - points[:, 1]
        c = points[:, 0]
        points_out = np.stack((r, c)).T
    else:
        points_out = None

    if image is not None:
        image_out = np.swapaxes(np.flipud(image), 0, 1)
    else:
        image_out = None

    return points_out, image_out


def rc_to_cart(points=None, deltao=None, image=None):
    """Convert image and/or points' coordinates from `r, c` to `x_c, y_c` convention.

        o--->....       .........
        |   c   .       ^ y_c   .
        |       .  ==>  |       .
        v r     .       |   x_c .
        .........       o--->....

    Parameters
    ----------
    points : (K, 2) ndarray, optional
        Array of points' `r, c` coordinates.
    deltao : int, optional
        Distance between `r` and `y_c` vectors' origins.
        Optional for `points` convertion if `image` is specified.
    image : (M, N[, C]) ndarray, optional
        Image stored in `x_c, y_c` axes convention.

    Returns
    -------
    points_out : (K, 2) ndarray
        Array of `points` `r, c` coordinates.
    image_out : (M, N[, C]) ndarray
        Same data as `image`, but in `r, c` convention.
    """
    if points is not None:
        if deltao is None:
            if image is None:
                msg = ('When converting coordinates, either `deltao` or `image` '
                       'have to be specified.')
                raise ValueError(msg)
            else:
                deltao = image.shape[0] - 1
        x_c = points[:, 1]
        y_c = deltao - points[:, 0]
        points_out = np.stack((x_c, y_c)).T
    else:
        points_out = None

    if image is not None:
        image_out = np.flipud(np.swapaxes(image, 0, 1))
    else:
        image_out = None

    return points_out, image_out


def xy_to_cart(points=None, deltao=None, image=None):
    """Convert image and/or points' coordinates from `x, y` to `x_c, y_c` convention.

        o--->....       .........
        |   x   .       ^ y_c   .
        |       .  ==>  |       .
        v y     .       |   x_c .
        .........       o--->....

    Parameters
    ----------
    points : (K, 2) ndarray, optional
        Array of points' `x, y` coordinates.
    deltao : int, optional
        Distance between `y` and `y_c` vectors' origins.
        Optional for `points` convertion if `image` is specified.
    image : (M, N[, C]) ndarray, optional
        Image stored in `x, y` axes convention.

    Returns
    -------
    points_out : (K, 2) ndarray
        Array of `points` `x_c, y_c` Cartesian coordinates.
    image_out : (M, N[, C]) ndarray
        Same data as `image`, but in `x_c, y_c` convention.
    """
    p, i = xy_to_rc(points, image)
    points_out, image_out = rc_to_cart(p, deltao, i)

    return points_out, image_out


def cart_to_xy(points=None, deltao=None, image=None):
    """Convert image and/or points' coordinates from `x_c, y_c` to `x, y` convention.

        .........       o--->....
        ^ y_c   .       |   x   .
        |       .  ==>  |       .
        |   x_c .       v y     .
        o--->....       .........

    Parameters
    ----------
    points : (K, 2) ndarray, optional
        Array of points' `x_c, y_c` Cartesian coordinates.
    deltao : int, optional
        Distance between `y_c` and `y` vectors' origins.
        Optional for `points` convertion if `image` is specified.
    image : (M, N[, C]) ndarray, optional
        Image stored in `x_c, y_y` axes convention.

    Returns
    -------
    points_out : (K, 2) ndarray
        Array of `points` `x, y` coordinates.
    image_out : (M, N[, C]) ndarray
        Same data as `image`, but in `x, y` convention.
    """
    p, i = cart_to_rc(points, deltao, image)
    points_out, image_out = rc_to_xy(p, i)

    return points_out, image_out

