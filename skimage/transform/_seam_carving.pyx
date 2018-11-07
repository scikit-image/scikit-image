# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as cnp


cdef cnp.double_t DBL_MAX = np.finfo(np.double).max

cdef void _preprocess_image(cnp.double_t[:, :, ::1] energy_img,
                            cnp.double_t[:, ::1] cumulative_img,
                            cnp.int8_t[:, ::1] track_img,
                            Py_ssize_t cols) nogil:
    """ For each row, compute the lowest seam value for all its columns.

    This function updates `cumulative_img` such that `cumulative_img[r, c]`
    is the total energy of the lowest energy seam ending at `(r, c)`.

    Parameters
    ----------
    energy_img : (M, N, 1) ndarray
        Cost array representing the expense to remove each pixel. Seam carving
        tries to avoid pixels with high costs.
    cumulative_img : (M, N) ndarray
        The array to be updated inplace with the total cost of lowest energy
        seams.
    track_img : (M, N) ndarray
        For each pixel, `track_img` stores the relative column offset in
        the previous row which has the lowest value in `cumulative_img`. This
        helps in in re-tracing the minimum cost seam.
    cols : int
        Number of columns to process.
    """

    cdef Py_ssize_t r, c, offset, c_idx
    cdef Py_ssize_t rows = energy_img.shape[0]
    cdef cnp.double_t min_cost = DBL_MAX
    cdef Py_ssize_t colsm1 = cols - 1
    cdef Py_ssize_t rm1

    for c in range(cols):
        cumulative_img[0, c] = energy_img[0, c, 0]

    for r in range(1, rows):
        rm1 = r - 1
        for c in range(cols):
            min_cost = DBL_MAX
            for offset in range(-1, 2):

                c_idx = c + offset
                if (c_idx > colsm1) or (c_idx < 0):
                    continue

                if cumulative_img[rm1, c_idx] < min_cost:
                    min_cost = cumulative_img[rm1, c_idx]
                    track_img[r, c] = offset

            cumulative_img[r, c] = min_cost + energy_img[r, c, 0]

cdef bint _mark_seam(cnp.int8_t[:, ::1] track_img,
                     Py_ssize_t start_index,
                     cnp.uint8_t[:, ::1] seam_map,
                     Py_ssize_t[::1] seam_buffer) nogil:

    """ Re-trace the optimal seam from a given column in the last row.

    This function tries to re-track an optimal seam from `start_index` and
    tries to mark it in `seam_map`. If this seam intersects with any existing
    seam in `seam_map` the function returns `0` without marking anything. Else
    it marks the seam in `seam_map` and returns `1`.

    track_img : (M, N) ndarray
       The array of relative column indices as updated by `_preprocess_image`.
    start_index : int
       The column number of the bottom most row from where to start re-tracing
       the seam.
    seam_map : (M, N) ndarray
        The array used to mark seams. If a pixel is marked as as seam it is set
        to `1`, else `0`.
    seam_buffer : (M,) ndarray
        Buffer used to store the column indices of the seam currently being
        checked. This is preallocated to save time.

    Returns
    -------
    success : int
        `1` if seam was marked, `0` is seam intersects and was not marked.
    """
    cdef Py_ssize_t rows = track_img.shape[0]
    cdef Py_ssize_t[::1] current_seam_indices = seam_buffer
    cdef Py_ssize_t row, col
    cdef cnp.int8_t offset
    cdef Py_ssize_t seams

    current_seam_indices[rows - 1] = start_index
    for row in range(rows - 2, -1, -1):
        col = current_seam_indices[row + 1]
        offset = track_img[row, col]
        col = col + offset
        current_seam_indices[row] = col

        if seam_map[row, col]:
            return 0

    for row in range(rows):
        col = current_seam_indices[row]
        seam_map[row, col] = 1

    return 1

cdef void _remove_seam(cnp.double_t[:, :, ::1] img,
                       cnp.uint8_t[:, ::1] seam_map, Py_ssize_t cols) nogil:
    """ Remove marked seams from an image.

    Parameters
    ----------
    img : (M, N, P) ndarray
        Input image whose vertical seams are to be removed.
    seam_map : (M, N) ndarray
        Array with seams to be removed marked by non-zero entries.
    cols : int
        The number of columns to process.
    """
    cdef Py_ssize_t rows = img.shape[0]
    cdef Py_ssize_t channels = img.shape[2]
    cdef Py_ssize_t r, c, ch, shift
    cdef Py_ssize_t c_shift

    for r in range(rows):
        shift = 0
        for c in range(cols):
            shift += seam_map[r, c]
            c_shift = c + shift
            for ch in range(channels):
                img[r, c, ch] = img[r, c_shift, ch]


def _seam_carve_v(img, energy_map, iters, border):
    """ Carve vertical seams off an image.

    Carves out vertical seams from an image while using the given energy map to
    decide the importance of each pixel.[1]_

    Parameters
    ----------
    img : (M, N) or (M, N, 3) ndarray
        Input image whose vertical seams are to be removed.
    energy_map : (M, N) ndarray
        Cost array denoting importance of each pixel. The algorithm will try to
        retain high valued pixels.
    iters : int
        Number of vertical seams to be removed.
    border : int, optional
        The number of pixels in the right, left and bottom end of the image
        to be excluded from being considered for a seam. This is important as
        certain filters just ignore image boundaries and set them to `0`.
        By default border is set to `1`.

    Returns
    -------
    image : (M, N - iters, 3) ndarray of float
        The cropped image with the vertical seams removed.

    References
    ----------
    .. [1] Shai Avidan and Ariel Shamir
           "Seam Carving for Content-Aware Image Resizing"
           https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Avidan07.pdf
    """
    # This reference has been kept to be used for the `np.argsort` call
    last_row_obj = np.zeros(img.shape[1], dtype=np.float)

    cdef cnp.double_t[::1] last_row = last_row_obj
    cdef Py_ssize_t[::1] sorted_indices
    cdef cnp.uint8_t[:, ::1] seam_map = np.zeros(img.shape[0:2],
                                                 dtype=np.uint8)
    cdef Py_ssize_t cols = img.shape[1]
    cdef Py_ssize_t rows = img.shape[0]
    cdef Py_ssize_t seams_left = iters
    cdef Py_ssize_t seams_removed
    cdef Py_ssize_t seam_idx
    cdef Py_ssize_t[::1] seam_buffer = np.zeros(rows, dtype=np.intp)

    cdef cnp.double_t[:, :, ::1] image = img
    cdef cnp.int8_t[:, ::1] track_img = np.zeros(img.shape[0:2], dtype=np.int8)
    cdef cnp.double_t[:, ::1] cumulative_img = np.zeros(img.shape[0:2],
                                                        dtype=np.float)
    cdef cnp.double_t[:, :, ::1] energy_img

    energy_map[:, 0:border] = DBL_MAX
    energy_map[:, cols-border:cols] = DBL_MAX

    # Filters often let the boundary be `0`. If all the entries in the last
    # row of `energy_img` are equal, the minimum value in the penultimate row
    # of `cumulative_img` will result in 3 minimum values in its last row.
    # Hence, two successive removals will always intersect as the 3 least seams
    # will share the same pixels except they will differ in the last row.
    energy_map[rows-border:rows, :] = energy_map[rows-2*border:rows-border, :]

    energy_map = np.ascontiguousarray(energy_map[:, :, np.newaxis])
    energy_img = energy_map

    _preprocess_image(energy_img, cumulative_img, track_img, cols)
    last_row[...] = cumulative_img[rows - 1, :]
    sorted_indices = np.argsort(last_row_obj)
    seam_idx = 0

    while seams_left > 0:
        if _mark_seam(track_img, sorted_indices[seam_idx], seam_map,
                      seam_buffer):
            seams_left -= 1
            cols -= 1
            seam_idx += 1
        else:
            seam_idx = 0
            _remove_seam(image, seam_map, cols)
            _remove_seam(energy_img, seam_map, cols)
            seam_map[...] = 0
            _preprocess_image(energy_img, cumulative_img, track_img, cols)
            last_row[:cols] = cumulative_img[rows - 1, :cols]
            sorted_indices = np.argsort(last_row_obj)

    _remove_seam(image, seam_map, cols)

    return img[:, 0:cols]
