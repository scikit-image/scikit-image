# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp

from libc.math cimport fabs, sqrt, ceil, atan2, M_PI, INFINITY

from ..draw import circle_perimeter

from _skimage2._shared.interpolation cimport round

cnp.import_array()


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
    H : ndarray, shape (radius index, M + 2R, N + 2R)
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
    cdef cnp.float64_t incr
    cdef cnp.ndarray[ndim=1, dtype=cnp.intp_t] circle_x, circle_y

    cdef cnp.ndarray[ndim=3, dtype=cnp.float64_t] acc = \
         np.zeros((radius.size,
                   img.shape[0] + 2 * offset,
                   img.shape[1] + 2 * offset), dtype=np.float64)

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


def _hough_ellipse(cnp.ndarray img, Py_ssize_t threshold=4,
                   cnp.float64_t accuracy=1, Py_ssize_t min_size=4,
                   max_size=None):
    """Perform an elliptical Hough transform.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold: int, optional (default 4)
        Accumulator threshold value.
    accuracy : float64, optional (default 1)
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

    cdef Py_ssize_t[:, ::1] pixels = np.vstack(np.nonzero(img))

    cdef Py_ssize_t num_pixels = pixels.shape[1]
    cdef list acc = list()
    cdef list results = list()
    cdef cnp.float64_t bin_size = accuracy * accuracy

    cdef cnp.float64_t max_b_squared
    if max_size is None:
        if img.shape[0] < img.shape[1]:
            max_b_squared = np.round(0.5 * img.shape[0])
        else:
            max_b_squared = np.round(0.5 * img.shape[1])
        max_b_squared *= max_b_squared
    else:
        max_b_squared = max_size * max_size

    cdef Py_ssize_t p1, p2, p3, p1x, p1y, p2x, p2y, p3x, p3y
    cdef cnp.float64_t xc, yc, a, b, d, k, dx, dy
    cdef cnp.float64_t cos_tau_squared, b_squared, orientation

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
                        results.append((hist_max,  # Accumulator
                                        yc, xc,
                                        a, b,
                                        orientation))
                    acc = []

    return np.array(results, dtype=[('accumulator', np.intp),
                                    ('yc', np.float64),
                                    ('xc', np.float64),
                                    ('a', np.float64),
                                    ('b', np.float64),
                                    ('orientation', np.float64)])


def _hough_line(cnp.ndarray img,
                cnp.ndarray[ndim=1, dtype=cnp.float64_t] theta):
    """Perform a straight line Hough transform.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    theta : 1D ndarray of float64
        Angles at which to compute the transform, in radians.

    Returns
    -------
    H : (P, Q) ndarray of uint64
        Hough transform accumulator.
    theta : ndarray
        Angles at which the transform was computed, in radians.
    distances : ndarray
        Distance values.

    Notes
    -----
    The origin is the top left corner of the original image.
    X and Y axis are horizontal and vertical edges respectively.
    The distance is the minimal algebraic distance from the origin
    to the detected line.

    Examples
    --------
    Generate a test image:

    >>> img = np.zeros((100, 150), dtype=bool)
    >>> img[30, :] = 1
    >>> img[:, 65] = 1
    >>> img[35:45, 35:50] = 1
    >>> for i in range(90):
    ...     img[i, i] = 1
    >>> rng = np.random.default_rng()
    >>> img += rng.random(img.shape) > 0.95

    Apply the Hough transform:

    >>> out, angles, d = hough_line(img)

    .. plot:: hough_tf.py

    """
    # Compute the array of angles and their sine and cosine
    cdef cnp.ndarray[ndim=1, dtype=cnp.float64_t] ctheta
    cdef cnp.ndarray[ndim=1, dtype=cnp.float64_t] stheta

    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    # compute the bins and allocate the accumulator array
    cdef cnp.ndarray[ndim=2, dtype=cnp.uint64_t] accum
    cdef cnp.ndarray[ndim=1, dtype=cnp.float64_t] bins
    cdef Py_ssize_t diag_len, rho0_idx, n_rhos

    # Length of diagonal from origin to bottom left of image.
    diag_len = <Py_ssize_t>ceil(sqrt(img.shape[0] * img.shape[0] +
                                     img.shape[1] * img.shape[1]))

    # Distance (rho) can be positive or negative.  Central index of the rho
    # axis corresponds to 0 rho (distance).
    rho0_idx = diag_len

    n_rhos = 2 * diag_len + 1  # Length of d axis of accumulator.
    accum = np.zeros((n_rhos, theta.shape[0]), dtype=np.uint64)
    # Distances corresponding to each element of d axis.
    bins = np.linspace(-diag_len, diag_len, n_rhos)

    # compute the nonzero indexes
    cdef cnp.ndarray[ndim=1, dtype=cnp.npy_intp] x_idxs, y_idxs
    y_idxs, x_idxs = np.nonzero(img)

    # finally, run the transform
    cdef Py_ssize_t nidxs, nthetas, i, j, x, y, accum_idx

    nidxs = y_idxs.shape[0]  # x and y are the same shape
    nthetas = theta.shape[0]
    with nogil:
        for i in range(nidxs):  # For every non-zero point in image.
            x = x_idxs[i]  # Fetch coordinate of point.
            y = y_idxs[i]
            for j in range(nthetas):  # For every theta.
                # Determine corresponding distance index for line going through
                # point.
                accum_idx = round((ctheta[j] * x + stheta[j] * y)) + rho0_idx
                accum[accum_idx, j] += 1

    return accum, theta, bins


def _probabilistic_hough_line(cnp.ndarray img, Py_ssize_t threshold,
                              Py_ssize_t line_length, Py_ssize_t line_gap,
                              cnp.ndarray[ndim=1, dtype=cnp.float64_t] theta,
                              rng=None,
                              Py_ssize_t lines_max=2 ** 15):
    """Return lines from a progressive probabilistic line Hough transform.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold : int
        Threshold in the accumulator to detect lines against noise.
    line_length : int
        Minimum accepted length of detected lines.
        Increase the parameter to extract longer lines.
    line_gap : int
        Maximum gap between pixels to still form a line.
        Increase the parameter to merge broken lines more aggressively.
    theta : (K,) ndarray of float64
        Angles at which to compute the transform, in radians.
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.
    lines_max : int, optional
        Maximum number of detectable lines.  Once this threshold is reached,
        further lines are discarded.

    Returns
    -------
    lines : list
        List of lines identified, lines in format ((x0, y0), (x1, y1)),
        indicating line start and end.

    References
    ----------
    .. [1] C. Galamhos, J. Matas and J. Kittler, "Progressive probabilistic
           Hough transform for line detection", in IEEE Computer Society
           Conference on Computer Vision and Pattern Recognition, 1999.

    Notes
    -----

    The algorithm (from [1]_) is the following:

    1. Check the (binarized) input image `img`; if it is empty then finish.
    2. Update the accumulator with a single pixel randomly selected from the
       input image.
    3. Remove pixel from input image.
    4. Check if the highest peak in the accumulator that was modified by the
       new pixel is higher than `threshold`. If not then goto 1.
    5. The peak in the accumulator from 4 corresponds to a line. Look along a
       corridor specified by that line, and find the longest segment of pixels
       either continuous or exhibiting a gap not exceeding a given threshold
       (`line_gap`). (In our implementation we use the nearest pixel coordinate
       on the line as the "corridor").
    6. Remove the pixels in the segment from input image.
    7. Unvote from the accumulator all the pixels from the line that have
       previously voted.
    8. If the line segment is longer than the minimum length (`line_length`)
       add it into the output list.
    9. goto 1.

    The code for this function started as a port of the OpenCV `hough.cpp`_
    file, copyright::

      2000 Intel Corporation, all rights reserved.
      2013 OpenCV Foundation, all rights reserved.
      2014, Itseez, Inc, all rights reserved.
      Third party copyrights are property of their respective owners.

    and released under a BSD-3-Clause license.

    .. _hough.cpp: https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/hough.cpp#L490
    """
    # Mask defines pixels still to be considered in the algorithm.
    # 1 for still-to-be-processed.
    cdef int PENDING = 1
    # 2 for processed, but not yet included in discovered line.  In this case
    # it has voted in the accumulator, and will need to be unvoted, if it
    # becomes part of a discovered line.  See STEP 7 above.
    cdef int VOTED = 2
    # 0 for absent, or fully-processed.
    cdef int CLEARED = 0
    cdef Py_ssize_t height = img.shape[0]
    cdef Py_ssize_t width = img.shape[1]
    # Allocate mask array.
    cdef cnp.uint8_t[:, ::1] mask = np.array(img != 0,
                                             dtype=np.uint8,
                                             order='C')

    # Order in which we will consider pixels (will be random).
    cdef cnp.intp_t[::1] rand_idxs
    cdef Py_ssize_t diag_len, n_rhos, rho0_idx, idx
    cdef Py_ssize_t j, x, y, x1, y1, px, py, rho_idx, max_theta_idx
    cdef Py_ssize_t reverse, gap, x_len, y_len, n_pts
    cdef cnp.float64_t line_sin, line_cos, rho, slope
    cdef cnp.int64_t value, max_value
    cdef int dx_is_1, delta, offset, slope_delta
    # Starting and ending x and y coordinates of current discovered line.
    cdef cnp.intp_t[:, ::1] line_ends = np.zeros((2, 2), dtype=np.intp)
    # Number of found pixels in current discovered line (not including gaps).
    cdef int n_line_pixels
    cdef Py_ssize_t nlines = 0  # The number of currently discovered lines.
    # Currently discovered lines.
    cdef cnp.intp_t[:, :, ::1] lines = np.zeros((lines_max, 2, 2),
                                                dtype=np.intp)
    diag_len = <Py_ssize_t>ceil((sqrt(img.shape[0] * img.shape[0] +
                                      img.shape[1] * img.shape[1])))
    # Maximum rho is maximum possible rho (distance) from the origin to the
    # closest point on a candidate line.  It cannot be greater than the length
    # of the diagonal from the origin to the bottom left of the image.
    n_rhos = diag_len * 2 + 1
    # Assemble n_rhos by n_thetas accumulator array.
    cdef cnp.int64_t[:, ::1] accum = np.zeros((n_rhos,
                                               theta.shape[0]),
                                              dtype=np.int64)
    # Distance (rho) can be positive or negative.  Central index of the rho
    # axis corresponds to 0 rho (distance).
    rho0_idx = diag_len

    cdef Py_ssize_t nthetas = theta.shape[0]

    # Diagonal length is also the longest possible line in the image.
    cdef cnp.intp_t[:, ::1] line_pixels = np.zeros((diag_len, 2),
                                                   dtype=np.intp)

    # compute sine and cosine of angles
    cdef cnp.float64_t[::1] ctheta = np.cos(theta)
    cdef cnp.float64_t[::1] stheta = np.sin(theta)

    # Find the nonzero indices.
    cdef cnp.intp_t[:] y_idxs, x_idxs
    y_idxs, x_idxs = np.nonzero(img)

    n_pts = len(x_idxs)
    if n_pts == 0:
        return []


    # Specify random order in which points will be processed.
    rng = np.random.default_rng(rng)
    rand_idxs = np.arange(n_pts, dtype=np.intp)
    rng.shuffle(rand_idxs)

    with nogil:
        for p_i in range(n_pts):
            # STEP 1: Select random non-zero point (see Notes).
            idx = rand_idxs[p_i]
            x = x_idxs[idx]
            y = y_idxs[idx]

            # Skip if eliminated by detection in earlier line search.
            if mask[y, x] == CLEARED:
                continue

            value = 0
            max_value = -1  # Max value in accumulator, start value.
            max_theta_idx = -1  # Index into {c,s}theta arrays, start value.

            # STEP 2: Apply Hough transform on point.
            for j in range(nthetas):
                rho = ctheta[j] * x + stheta[j] * y
                rho_idx = round(rho) + rho0_idx
                accum[rho_idx, j] += 1
                value = accum[rho_idx, j]
                if value > max_value:
                    max_value = value
                    max_theta_idx = j

            # STEP 3.  Note that this point has been processed, and voted on.
            mask[y, x] = VOTED

            if max_value < threshold:  # STEP 4.
                continue

            # STEP 5: From the random point (x, y), walk in opposite
            # directions and find line beginning and end.
            line_sin = stheta[max_theta_idx]
            line_cos = ctheta[max_theta_idx]
            # Line equation is rho = cos(theta) x + sin(theta) y.  Rearranging:
            # y = rho / sin(theta) - cos(theta) x / sin(theta), and slope
            # is -cos(theta) / sin(theta).
            # An abs slope of < 1 means that x increases more slowly than
            # y, and we should proceed in steps of 1 in x.  Otherwise
            # y increases more slowly than x, and we proceed in steps of
            # 1 in y.
            slope = INFINITY if line_sin == 0 else -line_cos / line_sin
            dx_is_1 = fabs(slope) < 1  # Should x advance in steps of 1?
            if not dx_is_1:  # abs(line_sin) <= abs(line_cos)
                slope = line_sin / -line_cos  # y advances in steps of 1.
            # Pass 1: identify pixels in line.  We walk through the line,
            # merging lines less than specified gap length.
            line_pixels[0, 0] = x  # Insert current point into pixel store.
            line_pixels[0, 1] = y
            line_ends[:, 0] = x
            line_ends[:, 1] = y
            n_line_pixels = 1
            for reverse in range(2):  # Forward and backward.
                gap = 0
                px = x
                py = y
                delta = -1 if reverse else 1
                offset = delta
                while True:
                    slope_delta = round(offset * slope)
                    if dx_is_1:
                        px = x + offset
                        py = y + slope_delta
                    else:  # Delta y is 1.
                        py = y + offset
                        px = x + slope_delta
                    # check when line exits image boundary
                    if px < 0 or px >= width or py < 0 or py >= height:
                        break
                    gap += 1
                    if mask[py, px] != CLEARED:
                        # Hit remaining pixel, continue line.
                        gap = 0
                        line_ends[reverse, 0] = px
                        line_ends[reverse, 1] = py
                        # Record presence of in-mask pixel on line.
                        line_pixels[n_line_pixels, 0] = px
                        line_pixels[n_line_pixels, 1] = py
                        n_line_pixels += 1
                    elif gap > line_gap:  # Gap to here too large, end line.
                        break
                    offset += delta

            # Confirm line length is sufficient.
            x_len = line_ends[1, 0] - line_ends[0, 0]  # pass 2 x - pass 1 x
            y_len = line_ends[1, 1] - line_ends[0, 1]  # pass 2 y - pass 1 y
            if sqrt(x_len * x_len + y_len * y_len) < line_length:
                continue

            # Pass 2: process detected pixels.
            for i in range(n_line_pixels):
                x1 = line_pixels[i, 0]
                y1 = line_pixels[i, 1]
                # STEP 7: remove any votes in accumulator from pixel in line
                # that we have collected votes for.
                if mask[y1, x1] == VOTED:
                    for j in range(nthetas):  # Remove accumulator votes.
                        rho = ctheta[j] * x1 + stheta[j] * y1
                        rho_idx = <int>round(rho) + rho0_idx
                        accum[rho_idx, j] -= 1
                # STEP 6: remove pixel from further consideration.
                mask[y1, x1] = CLEARED

            # STEP 8: Add line to the result.
            lines[nlines] = line_ends
            nlines += 1
            if nlines >= lines_max:
                break

    return [((line[0, 0], line[0, 1]), (line[1, 0], line[1, 1]))
            for line in lines[:nlines]]
