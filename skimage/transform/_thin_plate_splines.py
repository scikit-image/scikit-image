import numpy as np
import scipy as sp
from .._shared.utils import check_nD


class TpsTransform:
    """Thin-plate splines transformation.

    Apply thin-plate spline transformation between a set of control points.
    It interpolates a surface that passes through each control point.
    Control points are position constraints on a bending surface. The ideal
    surface is the one that bends least.

    Attributes
    ----------
    parameters : (N, D) array_like
        spline_mappings for every control point.
    src : (N, 2) array_like
        Coordinates of control points in source image.

    References
    ----------
    .. [1] Bookstein, Fred L. "Principal warps: Thin-plate splines and the
           decomposition of deformations," IEEE Transactions on pattern analysis
           and machine intelligence 11.6 (1989): 567–585. DOI:`10.1109/34.24792`

    Examples
    --------
    >>> import skimage as ski

    Define source and destination control points:

    >>> src = np.array([[0, 0], [0, 5], [5, 5],[5, 0]])
    >>> dst = np.roll(src, 1, axis=0)

     Generate meshgrid:

    >>> coords = np.meshgrid(np.arange(5), np.arange(5))
    >>> t_coords = np.vstack([coords[0].ravel(), coords[1].ravel()]).T

     Estimate transformation:

    >>> tps = ski.transform.TpsTransform()
    >>> tps.estimate(src, dst)
    True

    Apply the transformation:

    >>> trans_coord = tps(t_coords)
    >>> xx_trans = trans_coord[:, 0]
    >>> yy_trans = trans_coord[:, 1]
    >>> coords[1]
    array([[0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]])
    >>> expected_yy = np.array([0, 1, 2, 3, 4,
    ...                         0, 1, 2, 3, 4,
    ...                         0, 1, 2, 3, 4,
    ...                         0, 1, 2, 3, 4,
    ...                         0, 1, 2, 3, 4])
    >>> np.allclose(yy_trans, expected_yy)
    True
    """

    def __init__(self):
        self._estimated = False
        self.spline_mappings = None
        self.src = None

    def __call__(self, coords):
        """Estimate the transformation from a set of corresponding points.

        Parameters
        ----------
        coords : (N, 2) array_like
            x, y coordinates to transform

        Returns
        -------
        transformed_coords: (N, D) array
            Destination coordinates
        """
        if self.spline_mappings is None:
            raise ValueError(f"{self.spline_mappings}. Compute the `estimate`")
        coeffs = self.spline_mappings
        coords = np.array(coords)

        if not coords.ndim == 2 or coords.shape[1] != 2:
            raise ValueError("Input `coords` must have shape (N, 2)")

        radial_dist = self._radial_distance(coords[:, 0], coords[:, 1])

        x_warp = self._spline_function(
            coords[:, 0], coords[:, 1], radial_dist, coeffs[:, 0]
        )
        y_warp = self._spline_function(
            coords[:, 0], coords[:, 1], radial_dist, coeffs[:, 1]
        )
        return np.vstack([x_warp, y_warp]).T

    @property
    def inverse(self):
        raise NotImplementedError("This is yet to be implemented.")

    def estimate(self, src, dst):
        """Estimate optimal spline_mappings that describes the deformation of the points.


        Parameters
        ----------
        src : (N, 2) array_like
            Control point at source coordinates
        dst : (N, 2) array_like
            Control point at destination coordinates

        Returns
        -------
        success: bool
            True indicates that the spline_mappings were successfully estimated.

        Notes
        -----
        -  The number of source and destination points must match (N).
        """

        check_nD(src, 2)
        check_nD(dst, 2)

        if len(src) < 3 or len(dst) < 3:
            raise ValueError(f"{src} points less than 3 is considered undefined.")

        if src.shape != dst.shape:
            raise ValueError(
                "The shape of source and destination coordinates must match."
            )

        self.src = src
        n, d = src.shape

        dist = sp.spatial.distance.cdist(self.src, self.src)
        K = _radial_basis_kernel(dist)
        P = np.hstack([np.ones((n, 1)), src])
        L = np.zeros((n + 3, n + 3), dtype=np.float32)
        L[:n, :n] = K
        L[:n, -3:] = P
        L[-3:, :n] = P.T
        V = np.concatenate([dst, np.zeros((d + 1, d))])
        self.spline_mappings = np.dot(np.linalg.inv(L), V)
        return True

    def _radial_distance(self, x, y):
        """Compute the radial distance."""
        Pi_x = self.src[:, 0]
        Pi_y = self.src[:, 1]

        dx = Pi_x[:, np.newaxis] - x
        dy = Pi_y[:, np.newaxis] - y
        r = np.sqrt(dx**2 + dy**2)
        radial_dist = _radial_basis_kernel(r)
        return radial_dist

    def _spline_function(self, x, y, radial_dist, coeffs):
        """Solve the spline function in the X and Y directions"""
        w = coeffs[:-3]
        a1, ax, ay = coeffs[-3:]
        summation = np.dot(w, radial_dist)
        return a1 + ax * x + ay * y + summation


def _radial_basis_kernel(r):
    """Compute basis kernel for thin-plate splines.

    Parameters
    ----------
    r : (4, N) ndarray
        Input array representing the norm distance between interlandmark
        distances for the source form and based on the (x,y) coordinates
        for each of these points.

    Returns
    -------
    U : (4, N) ndarray
        Calculated kernel function U.
    """
    _small = 1e-8  # Small value to avoid divide-by-zero
    r_sq = r**2
    U = np.where(r == 0.0, 0.0, r_sq * np.log(r_sq + _small))
    return U


def tps_warp(
    image, src, dst, *, output_region=None, interpolation_order=1, grid_scaling=1
):
    """Return an array of warped images.

    Define a thin-plate-spline warping transform that warps from the
    src to the dst, and then warp the given images by
    that transform.

    Parameters
    ----------
    image : ndarray
        Input image.
    src : (N, 2)
        Control points at source coordinates.
    dst : (N, 2)
        Control points at target coordinates.
    output_region : tuple of integers, optional
        The region ``(xmin, ymin, xmax, ymax)`` of the output
        image that should be produced. (Note: The region is inclusive, i.e.
        xmin <= x <= xmax)
    interpolation_order : int, optional
        If 1, use linear interpolation, otherwise use
        nearest-neighbor interpolation.
    grid_scaling : int, optional
        If grid_scaling is greater than 1, say x, then the transform is
        defined on a grid x times smaller than the output image region.
        Then the transform is bilinearly interpolated to the larger region.
        This is fairly accurate for values up to 10 or so.

    Returns
    -------
    warped : array_like
        The warped input image.


    References
    ----------
    .. [1] Bookstein, Fred L. "Principal warps: Thin-plate splines and the
           decomposition of deformations," IEEE Transactions on pattern analysis
           and machine intelligence 11.6 (1989): 567–585.

    Examples
    --------
    Produce a warped image rotated by 90 degrees counter-clockwise.

    >>> import skimage as ski
    >>> astronaut = ski.data.astronaut()
    >>> image = ski.color.rgb2gray(astronaut)
    >>> src = np.array([[0, 0], [0, 512], [512, 512],[512, 0]])
    >>> dst = np.array([[512, 0], [0, 0], [0, 512],[512, 512]])
    >>> output_region = (0, 0, image.shape[0], image.shape[1])
    >>> warped_image = ski.transform.tps_warp(
    ...     image, src, dst, output_region=output_region
    ... )

    """
    image = np.asarray(image)

    if image.size == 0:
        raise ValueError(f"Cannot warp empty image with dimensions {image.shape!r}")

    if image.ndim not in (2, 3):
        raise ValueError("Only 2D and 3D images are supported")

    if output_region is not None:
        if not isinstance(output_region, tuple) or len(output_region) != 4:
            raise ValueError("Output region should be a tuple of 4 values.")
    else:
        output_region = (0, 0, image.shape[0] - 1, image.shape[1] - 1)

    x_min, y_min, x_max, y_max = output_region

    x_steps = (x_max - x_min + 1) // grid_scaling
    y_steps = (y_max - y_min + 1) // grid_scaling

    if x_steps <= 0 or y_steps <= 0:
        RuntimeError(f"Invalid or empty `output_region`: {output_region}")

    xx, yy = np.mgrid[x_min : x_max : x_steps * 1j, y_min : y_max : y_steps * 1j]
    coords = np.vstack([xx.ravel(), yy.ravel()]).T

    tform = TpsTransform()
    tform.estimate(dst, src)
    transform = tform(coords)

    transform = transform.reshape((x_steps, y_steps, 2))
    transform = [transform[..., 0], transform[..., 1]]

    if grid_scaling != 1:
        # linearly interpolate the zoomed transform grid
        new_x, new_y = np.mgrid[x_min:x_max, y_min:y_max]
        # new_x, new_y = np.mgrid[x_min:x_max+1, y_min:y_max+1]
        x_indices = (x_steps - 1) * (new_x - x_min) / float(x_max - x_min)
        y_indices = (y_steps - 1) * (new_y - y_min) / float(y_max - y_min)

        x_indices = np.clip(x_indices, 0, x_steps - 1)
        y_indices = np.clip(y_indices, 0, y_steps - 1)

        transform_x = sp.ndimage.map_coordinates(transform[0], [x_indices, y_indices])
        transform_y = sp.ndimage.map_coordinates(transform[1], [x_indices, y_indices])
        transform = [transform_x, transform_y]
    if image.ndim == 2:
        warped_image = sp.ndimage.map_coordinates(
            image, transform, order=interpolation_order
        )
    else:  # RGB image
        channels = image.shape[-1]
        warped_channels = [
            sp.ndimage.map_coordinates(
                image[..., channel], transform, order=interpolation_order
            )[..., None]
            for channel in range(channels)
        ]
        warped_image = np.concatenate(warped_channels, axis=-1)

    return warped_image
