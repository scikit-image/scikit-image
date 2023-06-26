import numpy as np
import scipy as sp


def warp_images(
    from_points,
    to_points,
    images,
    output_region,
    interpolation_order=1,
    approximate_grid=2
):
    """Return an array of warped images.

    Define a thin-plate-spline warping transform that warps from the
    from_points to the to_points, and then warp the given images by
    that transform.

    Parameters
    ----------
    from_points: (N, 2) array_like
        Source image coordinates.
    to_points: (N, 2) array_like
        Target image coordinates.
    images: ndarray
        Array of nD images to be warped with the given warp transform.
    output_region: (1, 4) array
        The (xmin, ymin, xmax, ymax) region of the output
        image that should be produced. (Note: The region is inclusive, i.e.
        xmin <= x <= xmax)
    interpolation_order: int, optional
        If value is 1, then use linear interpolation else use
        nearest-neighbor interpolation.
    approximate_grid: int, optional
        If approximate_grid is greater than 1, say x, then the transform is
        defined on a grid x times smaller than the output image region.
        Then the transform is bilinearly interpolated to the larger region.
        This is fairly accurate for values up to 10 or so.

    Returns
    -------
    warped: array_like
        Array of warped images.

    Examples
    --------
    >>> import skimage as ski
    >>> image = ski.data.astronaut()
    >>> astronaut = ski.color.rgb2gray(image)
    >>> from_points = np.array([[0, 0], [0, 500], [500, 500],[500, 0]])
    >>> to_points = np.array([[500, 0], [0, 0], [0, 500],[500, 500]])
    >>> output_region = (0, 0, astronaut.shape[1], astronaut.shape[0])
    >>> warped_image = warp_images(from_points, to_points,
                        images=[astronaut], output_region=output_region)
    References
    ----------
    .. [1] Bookstein, Fred L. "Principal warps: Thin-plate splines and the
    decomposition of deformations." IEEE Transactions on pattern analysis and
    machine intelligence 11.6 (1989): 567–585.

    """
    transform = _make_inverse_warp(from_points, to_points,
                                   output_region, approximate_grid)
    return [sp.ndimage.map_coordinates(np.asarray(image), transform, order=interpolation_order) for image in images]


def _make_inverse_warp(
    from_points,
    to_points,
    output_region,
    approximate_grid
):
    """Compute inverse warp tranform.

    Parameters
    ----------
    from_points : (N,2) array_like
        An array of N points representing the source point.
    to_points : (N,2) array_like
        An array of N points representing the target point.
    output_region: (1, 4) array
        The (xmin, ymin, xmax, ymax) region of the output
        image that should be produced. (Note: The region is inclusive, i.e.
        xmin <= x <= xmax)
    interpolation_order: int, optional
        If value is 1, then use linear interpolation else use
        nearest-neighbor interpolation.
    approximate_grid: int, optional
        If approximate_grid is greater than 1, say x, then the transform is
        defined on a grid x times smaller than the output image region.
        Then the transform is bilinearly interpolated to the larger region.
        This is fairly accurate for values up to 10 or so.

    Returns
    -------
    warped: array_like
        Array of warped images.

    """
    x_min, y_min, x_max, y_max = output_region
    if approximate_grid is None:
        approximate_grid = 1
    x_steps = (x_max - x_min) // approximate_grid
    y_steps = (y_max - y_min) // approximate_grid
    x, y = np.mgrid[x_min:x_max:x_steps*1j, y_min:y_max:y_steps*1j]

    # make the reverse transform warping from the to_points to the from_points,
    # because we do image interpolation in this reverse fashion
    transform = _tps_transform(to_points, from_points, x, y)


    if approximate_grid != 1:
        # linearly interpolate the zoomed transform grid
        new_x, new_y = np.mgrid[x_min:x_max, y_min:y_max]
        # new_x, new_y = np.mgrid[x_min:x_max+1, y_min:y_max+1]
        x_indices = ((x_steps - 1) * (new_x - x_min) / float(x_max - x_min))
        y_indices = ((y_steps - 1) * (new_y - y_min) / float(y_max - y_min))

        x_indices = np.clip(x_indices, 0, x_steps - 1)
        y_indices = np.clip(y_indices, 0, y_steps - 1)



        transform_x = sp.ndimage.map_coordinates(transform[0], [x_indices, y_indices])
        transform_y = sp.ndimage.map_coordinates(transform[1], [x_indices, y_indices])
        transform = [transform_x, transform_y]
    return transform


def _U(x):
    """Compute basis kernel function for thine-plate splines.

    Parameters
    ----------
    x: ndarray
        Input array representing the norm distance between points.
        The norm is the Euclidean distance.
    Returns
    -------
    ndarray
        Calculated U values.
    """
    _small = 1e-8  # Small value to avoid divide-by-zero
    return np.where(x == 0.0, 0.0, (x**2) * np.log((x) + _small))

def _make_L_matrix(points):
    """Create a L matrix based on the given points.

    A (N+P+1, N+P+1)-shaped L matrix that gets inverted when calculating
    the thin-plate spline from `points`.

    Parameters
    ----------
    points : (N, 2) shaped array_like
        A (N, D) array of N point in D=2 dimensions.

    Returns
    -------
    L : ndarray
        A (N+D+1, N+D+1) shaped array of the form [[K | P][P.T | 0]].
    """
    if points.ndim != 2:
        raise ValueError("The input `points` must be a 2-D tensor")
    n_pts = points.shape[0]
    P = np.hstack([np.ones((n_pts, 1)), points])
    K = _U(sp.spatial.distance.cdist(points, points, metric='euclidean'))
    O = np.zeros((3, 3))
    L = np.asarray(np.bmat([[K, P], [P.transpose(), O]]))
    return L

def _coeffs(from_points, to_points):
    """Find the thin-plate spline coefficients.

    Parameters
    ----------
    from_points : (N, 2) array_like
        An array of N points representing the source point.
    to_points : (N,2) array_like
        An array of N points representing the target point.
        `to_points` must have the same shape as `from_points`.

    Returns
    -------
    coeffs : ndarray
        Array of shape (N+3, 2) containing the calculated coefficients.

    """
    n_coords = from_points.shape[1]
    Y = np.row_stack((to_points, np.zeros((n_coords+1, n_coords))))
    L = _make_L_matrix(from_points)
    coeffs = np.dot(np.linalg.pinv(L), Y)
    return coeffs


def _calculate_f(x, y, points, coeffs):
    """Compute the thin-plate spline function at given coordinates (x, y).

    Parameters:
    ----------
    coeffs : ndarray
        Array of shape (N+3, 2) containing the thin-plate spline coefficients.
    points : ndarray
        Array of shape (N, 2) representing the source point.
    x : ndarray
        Array representing the x-coordinates of points to transform.
    y : ndarray
        Array representing the y-coordinates of points to transform.

    Returns:
    -------
    ndarray :
        Array containing the computed thin-plate spline function values.

    Notes:
    -----
    This function calculates the thin-plate spline function values at each
    coordinate (x, y) everywhere in the plane.

    The function is calculated as:

    .. math::

        f(x, y) = a1 + ax * x + ay * y + Σ(wi * U(|Pi - (x, y)|))

        where:
        - a1, ax, and ay are the last three coefficients in `coeffs`.
        - wi represents the weights corresponding to each point Pi in `points`.
        - U(r) is the basis kernel function.

    """
    w = coeffs[:-3]
    a1, ax, ay = coeffs[-3:]
    summation = np.zeros(x.shape)
    for wi, Pi in zip(w, points):
        summation += wi * _U(np.sqrt((Pi[0]-x)**2 + (Pi[1]-y)**2))
    return a1 + ax*x + ay*y + summation


def _tps_transform(from_points, to_points, x_vals, y_vals):
    """Apply transformation to coordinates `x_vals` and `y_vals`.

    Parameters
    ----------
    from_points : (N,2) array_like
        An array of N points representing the source point.
    to_points : (N,2) array_like
        An array of N points representing the target point.
        `to_points` must have the same shape as `from_points`.
    x_vals : array_like
        The x-coordinates of points to transform.
    y_vals : array_like
        The y-coordinates of points to transform.

    Returns
    -------
    transformed_pts : lists
        A list of transformed coordinates.
    """
    from_points = np.asarray(from_points)
    to_points = np.asarray(to_points)
    # err = np.seterr(divide='ignore')
    coeffs = _coeffs(from_points, to_points)
    transformed_x = _calculate_f(x_vals, y_vals, from_points, coeffs[:, 0])
    transformed_y = _calculate_f(x_vals, y_vals, from_points, coeffs[:, 1])
    # np.seterr(**err)
    return [transformed_x, transformed_y]
