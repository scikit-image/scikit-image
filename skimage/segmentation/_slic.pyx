#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libc.float cimport DBL_MAX

import numpy as np
cimport numpy as cnp

from skimage.util import regular_grid

def _slic_cython(double[:, :, :, ::1] image_zyx,
                 double[:, ::1] segments,
                 Py_ssize_t max_iter,
                 double[::1] spacing):
    """Helper function for SLIC segmentation.

    Parameters
    ----------
    image_zyx : 4D array of double, shape (Z, Y, X, C)
        The input image.
    segments : 2D array of double, shape (N, 3 + C)
        The initial centroids obtained by SLIC as [Z, Y, X, C...].
    max_iter : int
        The maximum number of k-means iterations.
    spacing : 1D array of double, shape (3,)
        The voxel spacing along each image dimension. This parameter
        controls the weights of the distances along z, y, and x during
        k-means clustering.

    Returns
    -------
    nearest_segments : 3D array of int, shape (Z, Y, X)
        The label field/superpixels found by SLIC.

    Notes
    -----
    The image is considered to be in (z, y, x) order, which can be
    surprising. More commonly, the order (x, y, z) is used. However,
    in 3D image analysis, 'z' is usually the "special" dimension, with,
    for example, a different effective resolution than the other two
    axes. Therefore, x and y are often processed together, or viewed as
    a cut-plane through the volume. So, if the order was (x, y, z) and
    we wanted to look at the 5th cut plane, we would write::

        my_z_plane = img3d[:, :, 5]

    but, assuming a C-contiguous array, this would grab a discontiguous
    slice of memory, which is bad for performance. In contrast, if we
    see the image as (z, y, x) ordered, we would do::

        my_z_plane = img3d[5]

    and get back a contiguous block of memory. This is better both for
    performance and for readability.
    """

    # initialize on grid
    cdef Py_ssize_t depth, height, width
    depth = image_zyx.shape[0]
    height = image_zyx.shape[1]
    width = image_zyx.shape[2]

    cdef Py_ssize_t n_segments = segments.shape[0]
    # number of features [X, Y, Z, ...]
    cdef Py_ssize_t n_features = segments.shape[1]

    # approximate grid size for desired n_segments
    cdef Py_ssize_t step_z, step_y, step_x
    slices = regular_grid((depth, height, width), n_segments)
    step_z, step_y, step_x = [int(s.step) for s in slices]

    cdef Py_ssize_t[:, :, ::1] nearest_segments \
        = np.empty((depth, height, width), dtype=np.intp)
    cdef double[:, :, ::1] distance \
        = np.empty((depth, height, width), dtype=np.double)
    cdef Py_ssize_t[::1] n_segment_elems = np.zeros(n_segments, dtype=np.intp)

    cdef Py_ssize_t i, c, k, x, y, z, x_min, x_max, y_min, y_max, z_min, z_max
    cdef char change
    cdef double dist_center, cx, cy, cz, dy, dz

    cdef double sz, sy, sx
    sz = spacing[0]
    sy = spacing[1]
    sx = spacing[2]

    for i in range(max_iter):
        change = 0
        distance[:, :, :] = DBL_MAX

        # assign pixels to segments
        for k in range(n_segments):

            # segment coordinate centers
            cz = segments[k, 0]
            cy = segments[k, 1]
            cx = segments[k, 2]

            # compute windows
            z_min = <Py_ssize_t>max(cz - 2 * step_z, 0)
            z_max = <Py_ssize_t>min(cz + 2 * step_z + 1, depth)
            y_min = <Py_ssize_t>max(cy - 2 * step_y, 0)
            y_max = <Py_ssize_t>min(cy + 2 * step_y + 1, height)
            x_min = <Py_ssize_t>max(cx - 2 * step_x, 0)
            x_max = <Py_ssize_t>min(cx + 2 * step_x + 1, width)

            for z in range(z_min, z_max):
                dz = (sz * (cz - z)) ** 2
                for y in range(y_min, y_max):
                    dy = (sy * (cy - y)) ** 2
                    for x in range(x_min, x_max):
                        dist_center = dz + dy + (sx * (cx - x)) ** 2
                        for c in range(3, n_features):
                            dist_center += (image_zyx[z, y, x, c - 3]
                                            - segments[k, c]) ** 2
                        if distance[z, y, x] > dist_center:
                            # segments start at 1
                            nearest_segments[z, y, x] = k+1
                            distance[z, y, x] = dist_center
                            change = 1

        # stop if no pixel changed its segment
        if change == 0:
            break

        # recompute segment centers

        # sum features for all segments
        n_segment_elems[:] = 0
        segments[:, :] = 0
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    #compensate the label offset 1
                    k = nearest_segments[z, y, x] - 1
                    n_segment_elems[k] += 1
                    segments[k, 0] += z
                    segments[k, 1] += y
                    segments[k, 2] += x
                    for c in range(3, n_features):
                        segments[k, c] += image_zyx[z, y, x, c - 3]

        # divide by number of elements per segment to obtain mean
        for k in range(n_segments):
            for c in range(n_features):
                segments[k, c] /= n_segment_elems[k]

    return np.asarray(nearest_segments)


def _enforce_label_connectivity_cython(Py_ssize_t[:, :, ::1] segments,
                                Py_ssize_t n_segments,
                                int min_size,
                                int max_size):
    """ Helper function to remove small disconnected regions from the labels

    Parameters
    ----------
    segments : 3D array of int, shape (Z, Y, X)
        The label field/superpixels found by SLIC.
    n_segments: int
        number of specified segments
    min_size: int
        minimum size of the segment
    max_size: int
        maximum size of the segment. This is done for performance reasons,
        to pre-allocate a sufficiently large array for the breadth first search
    Returns
    -------
    connected_segments : 3D array of int, shape (Z, Y, X)
        A label field with connected labels starting at label=1
    """

    #get image dimensions
    cdef Py_ssize_t depth, height, width
    depth = segments.shape[0]
    height = segments.shape[1]
    width = segments.shape[2]

    #neighborhood arrays
    cdef Py_ssize_t[:] ddx = np.array((1,-1,0,0,0,0))
    cdef Py_ssize_t[:] ddy = np.array((0,0,1,-1,0,0))
    cdef Py_ssize_t[:] ddz = np.array((0,0,0,0,1,-1))

    #new object with connected segments
    cdef Py_ssize_t[:, :, ::1] connected_segments = np.zeros_like(segments)

    cdef Py_ssize_t current_new_label       = 0
    cdef Py_ssize_t label = 0

    #variables for the breadth first search
    cdef Py_ssize_t count = 1
    cdef Py_ssize_t p = 0
    cdef Py_ssize_t adjacent

    cdef Py_ssize_t zz,yy,xx

    cdef Py_ssize_t[:, ::1] coord_list = np.zeros((max_size,3), dtype=np.intp)

    #loop through all image
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if (connected_segments[z,y,x] > 0):
                    continue
                #find the component size
                adjacent = 0
                label = segments[z,y,x]
                current_new_label += 1
                connected_segments[z,y,x] = current_new_label

                count  = 1
                p = 0
                coord_list[p,0] = z
                coord_list[p,1] = y
                coord_list[p,2] = x

                #perform a breadth first search to find the size of the connected component
                while (p != count):
                    for i in range(6):
                        zz = coord_list[p,0] + ddz[i]
                        yy = coord_list[p,1] + ddy[i]
                        xx = coord_list[p,2] + ddx[i]
                        if (xx >= 0 and xx < width and yy >= 0 and yy < height and zz >= 0 and zz < depth):
                            if (segments[zz,yy,xx] == label and connected_segments[zz,yy,xx] == 0):
                                connected_segments[zz,yy,xx] = current_new_label
                                coord_list[count,0] = zz
                                coord_list[count,1] = yy
                                coord_list[count,2] = xx
                                count = count + 1
                            elif (connected_segments[zz,yy,xx] > 0 and connected_segments[zz,yy,xx] != current_new_label):
                                adjacent = connected_segments[zz,yy,xx]
                    p = p + 1


                #change to an adjacent one, like in the original paper
                if (count < min_size):
                    for i in range(count):
                        connected_segments[coord_list[i,0],coord_list[i,1],coord_list[i,2]] = adjacent

    return np.asarray(connected_segments)