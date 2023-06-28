import numpy as np
import scipy as sp


def tps_warp(
    image,
    src,
    dst,
    output_region,
    interpolation_order=1,
    grid_scaling=None
):
    """Return an array of warped images.

    Define a thin-plate-spline warping transform that warps from the
    src to the dst, and then warp the given images by
    that transform.

    Parameters
    ----------
    src: (N, 2) array_like
        Source image coordinates.
    dst: (N, 2) array_like
        Destination image coordinates.
    image: ndarray
        Input image.
    output_region: (1, 4) array
        The (xmin, ymin, xmax, ymax) region of the output
        image that should be produced. (Note: The region is inclusive, i.e.
        xmin <= x <= xmax)
    interpolation_order: int, optional
        If value is 1, then use linear interpolation else use
        nearest-neighbor interpolation.
    grid_scaling: int, optional
        If grid_scaling is greater than 1, say x, then the transform is
        defined on a grid x times smaller than the output image region.
        Then the transform is bilinearly interpolated to the larger region.
        This is fairly accurate for values up to 10 or so.

    Returns
    -------
    warped: array_like
        Array of warped images.

    Examples
    --------
    Produce a warped image rotated by 90 degrees counter-clockwise:

    >>> import skimage as ski
    >>> image = ski.data.astronaut()
    >>> astronaut = ski.color.rgb2gray(image)
    >>> src = np.array([[0, 0], [0, 500], [500, 500],[500, 0]])
    >>> dst = np.array([[500, 0], [0, 0], [0, 500],[500, 500]])
    >>> output_region = (0, 0, astronaut.shape[1], astronaut.shape[0])
    >>> warped_image = ski.transform.tps_warp(image, src, dst,
                                    output_region=output_region)
    References
    ----------
    .. [1] Bookstein, Fred L. "Principal warps: Thin-plate splines and the
    decomposition of deformations." IEEE Transactions on pattern analysis and
    machine intelligence 11.6 (1989): 567–585.

    """
    if image.size == 0:
        raise ValueError("Cannot warp empty image with dimensions",
                         image.shape)

    if image.ndim != 2:
        raise ValueError("Only 2-D images (grayscale or color) are "
                         "supported")

    transform = _make_inverse_warp(src, dst,
                                   output_region, grid_scaling)
    return sp.ndimage.map_coordinates(np.asarray(image), transform, order=interpolation_order)


def _make_inverse_warp(
    src,
    dst,
    output_region,
    grid_scaling,
):
    """Compute inverse warp tranform.

    Parameters
    ----------
    src : (N,2) array_like
        An array of N points representing the source coordinates.
    dst : (N,2) array_like
        An array of N points representing the destination coordinates.
    output_region: (1, 4) array
        The (xmin, ymin, xmax, ymax) region of the output
        image that should be produced. (Note: The region is inclusive, i.e.
        xmin <= x <= xmax)
    interpolation_order: int, optional
        If value is 1, then use linear interpolation else use
        nearest-neighbor interpolation.
    grid_scaling: int, optional
        If grid_scaling is greater than 1, say x, then the transform is
        defined on a grid x times smaller than the output image region.
        Then the transform is bilinearly interpolated to the larger region.
        This is fairly accurate for values up to 10 or so.

    Returns
    -------
    warped: array_like
        Array of warped images.

    """
    x_min, y_min, x_max, y_max = output_region
    if grid_scaling is None:
        grid_scaling = 1
    x_steps = (x_max - x_min) // grid_scaling
    y_steps = (y_max - y_min) // grid_scaling
    x, y = np.mgrid[x_min:x_max:x_steps*1j, y_min:y_max:y_steps*1j]

    # make the reverse transform warping from the dst to the src,
    # because we do image interpolation in this reverse fashion
    transform = tps_transform(dst, src, x, y)


    if grid_scaling != 1:
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
    points : (N, 2) array_like
        A (N, D) array of N point in D=2 dimensions.

    Returns
    -------
    L : ndarray
        A (N+D+1, N+D+1) shaped array of the form [[K | P][P.T | 0]].
    """
    n_pts = points.shape[0]
    P = np.hstack([np.ones((n_pts, 1)), points])
    K = _U(sp.spatial.distance.cdist(points, points, metric='euclidean'))
    O = np.zeros((3, 3))
    L = np.asarray(np.bmat([[K, P], [P.transpose(), O]]))
    return L

def _coeffs(src, dst):
    """Find the thin-plate spline coefficients.

    Parameters
    ----------
    src : (N, 2) array_like
        An array of N points representing the source point.
    dst : (N,2) array_like
        An array of N points representing the destination point.
        `dst` must have the same shape as `src`.

    Returns
    -------
    coeffs : ndarray
        Array of shape (N+3, 2) containing the calculated coefficients.

    """
    n_coords = src.shape[1]
    Y = np.row_stack((dst, np.zeros((n_coords+1, n_coords))))
    L = _make_L_matrix(src)
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


def tps_transform(src, dst, x_vals, y_vals):
    """Apply transformation to coordinates `x_vals` and `y_vals`.

    Parameters
    ----------
    src : (N, 2) array_like
        An array of N points representing the source point.
    dst : (N, 2) array_like
        An array of N points representing the destination point.
        `dst` must have the same shape as `src`.
    x_vals : array_like
        The x-coordinates of points to transform.
    y_vals : array_like
        The y-coordinates of points to transform.

    Returns
    -------
    transformed_pts : lists
        A list of transformed coordinates.

    Examples
    --------
    >>> import skimage as ski

    Define source and destination points:

    >>> src = np.array([[0, 0], [0, 512], [512, 512],[512, 0]])
    >>> dst = np.roll(src_points, 1, axis=0)

    Generate the grid points for transformation

    >>> x_min, x_max = 0, 90
    >>> y_min, y_max = 0, 90
    >>> num_points = 100
    >>> x, y = np.meshgrid(np.linspace(x_min, x_max, num_points),
                   np.linspace(y_min, y_max, num_points))

    Apply the transformation

    >>> transformed_x, transformed_y = ski.transform.tps_transform(
            src_points, dst_points, x, y)
    >>> transformed_x[10, 10]
    501.88594868313635
    >>> transformed_y[10, 10]
    10.112359869775215
    """
    src = np.asarray(src)
    dst = np.asarray(dst)

    if src.shape != dst.shape:
        raise ValueError("The `src` and `dst` must be of the same shape.")

    if src.shape[1] != 2 or dst.shape[1] != 2:
        raise ValueError("The input `src` or `dst` must have shape (N, 2)")

    # err = np.seterr(divide='ignore')
    coeffs = _coeffs(src, dst)
    transformed_x = _calculate_f(x_vals, y_vals, src, coeffs[:, 0])
    transformed_y = _calculate_f(x_vals, y_vals, src, coeffs[:, 1])
    # np.seterr(**err)
    return [transformed_x, transformed_y]
