#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp

from libc.math cimport sqrt, atan2, M_PI

from ..draw import circle_perimeter


def _hough_circle(cnp.ndarray img,
                  cnp.ndarray[ndim=1, dtype=cnp.intp_t] radius,
                  char normalize=True, char full_output=False):
    """Perform a circular Hough transform.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    radius : ndarray
        Radii at which to compute the Hough transform.
    normalize : boolean, optional (default True)
        Normalize the accumulator with the number
        of pixels used to draw the radius.
    full_output : boolean, optional (default False)
        Extend the output size by twice the largest
        radius in order to detect centers outside the
        input picture.

    Returns
    -------
    H : 3D ndarray (radius index, (M + 2R, N + 2R) ndarray)
        Hough transform accumulator for each radius.
        R designates the larger radius if full_output is True.
        Otherwise, R = 0.
    """
    if img.ndim != 2:
        raise ValueError('The input image must be 2D.')

    cdef Py_ssize_t xmax = img.shape[0]
    cdef Py_ssize_t ymax = img.shape[1]

    # compute the nonzero indexes
    cdef cnp.ndarray[ndim=1, dtype=cnp.intp_t] x, y
    x, y = np.nonzero(img)

    cdef Py_ssize_t num_pixels = x.size

    cdef Py_ssize_t offset = 0
    if full_output:
        # Offset the image
        offset = radius.max()
        x = x + offset
        y = y + offset

    cdef Py_ssize_t i, p, c, num_circle_pixels, tx, ty
    cdef double incr
    cdef cnp.ndarray[ndim=1, dtype=cnp.intp_t] circle_x, circle_y

    cdef cnp.ndarray[ndim=3, dtype=cnp.double_t] acc = \
         np.zeros((radius.size,
                   img.shape[0] + 2 * offset,
                   img.shape[1] + 2 * offset), dtype=np.double)

    for i, rad in enumerate(radius):
        # Store in memory the circle of given radius
        # centered at (0,0)
        circle_x, circle_y = circle_perimeter(0, 0, rad)

        num_circle_pixels = circle_x.size

        with nogil:

            if normalize:
                incr = 1.0 / num_circle_pixels
            else:
                incr = 1

            # For each non zero pixel
            for p in range(num_pixels):
                # Plug the circle at (px, py),
                # its coordinates are (tx, ty)
                for c in range(num_circle_pixels):
                    tx = circle_x[c] + x[p]
                    ty = circle_y[c] + y[p]
                    if offset:
                        acc[i, tx, ty] += incr
                    elif 0 <= tx < xmax and 0 <= ty < ymax:
                        acc[i, tx, ty] += incr

    return acc


def _hough_ellipse(cnp.ndarray img, Py_ssize_t threshold=4, double accuracy=1,
                   Py_ssize_t min_size=4, max_size=None):
    """Perform an elliptical Hough transform.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold: int, optional (default 4)
        Accumulator threshold value.
    accuracy : double, optional (default 1)
        Bin size on the minor axis used in the accumulator.
    min_size : int, optional (default 4)
        Minimal major axis length.
    max_size : int, optional
        Maximal minor axis length. (default None)
        If None, the value is set to the half of the smaller
        image dimension.

    Returns
    -------
    result : ndarray with fields [(accumulator, yc, xc, a, b, orientation)]
          Where ``(yc, xc)`` is the center, ``(a, b)`` the major and minor
          axes, respectively. The `orientation` value follows
          `skimage.draw.ellipse_perimeter` convention.

    Examples
    --------
    >>> img = np.zeros((25, 25), dtype=np.uint8)
    >>> rr, cc = ellipse_perimeter(10, 10, 6, 8)
    >>> img[cc, rr] = 1
    >>> result = hough_ellipse(img, threshold=8)
    [(10, 10.0, 8.0, 6.0, 0.0, 10.0)]

    Notes
    -----
    The accuracy must be chosen to produce a peak in the accumulator
    distribution. In other words, a flat accumulator distribution with low
    values may be caused by a too low bin size.

    References
    ----------
    .. [1] Xie, Yonghong, and Qiang Ji. "A new efficient ellipse detection
           method." Pattern Recognition, 2002. Proceedings. 16th International
           Conference on. Vol. 2. IEEE, 2002
    """
    if img.ndim != 2:
            raise ValueError('The input image must be 2D.')

    # The creation of the array `pixels` results in a rather nasty error
    # when the image is empty.
    # As discussed in GitHub #2820 and #2996, we opt to return an empty array.
    if not np.any(img):
        return np.zeros((0, 6))

    cdef Py_ssize_t[:, ::1] pixels = np.row_stack(np.nonzero(img))

    cdef Py_ssize_t num_pixels = pixels.shape[1]
    cdef list acc = list()
    cdef list results = list()
    cdef double bin_size = accuracy * accuracy

    cdef double max_b_squared
    if max_size is None:
        if img.shape[0] < img.shape[1]:
            max_b_squared = np.round(0.5 * img.shape[0])
        else:
            max_b_squared = np.round(0.5 * img.shape[1])
        max_b_squared *= max_b_squared
    else:
        max_b_squared = max_size * max_size

    cdef Py_ssize_t p1, p2, p3, p1x, p1y, p2x, p2y, p3x, p3y
    cdef double xc, yc, a, b, d, k, dx, dy
    cdef double cos_tau_squared, b_squared, orientation

    for p1 in range(num_pixels):
        p1x = pixels[1, p1]
        p1y = pixels[0, p1]

        for p2 in range(p1):
            p2x = pixels[1, p2]
            p2y = pixels[0, p2]

            # Candidate: center (xc, yc) and main axis a
            dx = p1x - p2x
            dy = p1y - p2y
            a = 0.5 * sqrt(dx * dx + dy * dy)
            if a > 0.5 * min_size:
                xc = 0.5 * (p1x + p2x)
                yc = 0.5 * (p1y + p2y)

                for p3 in range(num_pixels):
                    p3x = pixels[1, p3]
                    p3y = pixels[0, p3]
                    dx = p3x - xc
                    dy = p3y - yc
                    d = sqrt(dx * dx + dy * dy)
                    if d > min_size:
                        dx = p3x - p1x
                        dy = p3y - p1y
                        cos_tau_squared = ((a*a + d*d - dx*dx - dy*dy)
                                           / (2 * a * d))
                        cos_tau_squared *= cos_tau_squared
                        # Consider b2 > 0 and avoid division by zero
                        k = a*a - d*d * cos_tau_squared
                        if k > 0 and cos_tau_squared < 1:
                            b_squared = a*a * d*d * (1 - cos_tau_squared) / k
                            # b2 range is limited to avoid histogram memory
                            # overflow
                            if b_squared <= max_b_squared:
                                acc.append(b_squared)

                if len(acc) > 0:
                    bins = np.arange(0, np.max(acc) + bin_size, bin_size)
                    hist, bin_edges = np.histogram(acc, bins=bins)
                    hist_max = np.max(hist)
                    if hist_max > threshold:
                        orientation = atan2(p1x - p2x, p1y - p2y)
                        b = sqrt(bin_edges[hist.argmax()])
                        # to keep ellipse_perimeter() convention
                        if orientation != 0:
                            orientation = M_PI - orientation
                            # When orientation is not in [-pi:pi]
                            # it would mean in ellipse_perimeter()
                            # that a < b. But we keep a > b.
                            if orientation > M_PI:
                                orientation = orientation - M_PI / 2.
                                a, b = b, a
                        results.append((hist_max, # Accumulator
                                        yc, xc,
                                        a, b,
                                        orientation))
                    acc = []

    return np.array(results, dtype=[('accumulator', np.intp),
                                    ('yc', np.double),
                                    ('xc', np.double),
                                    ('a', np.double),
                                    ('b', np.double),
                                    ('orientation', np.double)])
