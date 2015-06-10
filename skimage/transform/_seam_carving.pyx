import numpy as np
cimport numpy as cnp


cdef cnp.double_t DBL_MAX = np.finfo(np.double).max

cdef _preprocess_image(cnp.double_t[:, :, ::1] energy_img,
                       cnp.double_t[:, ::1] cumulative_img,
                       cnp.int8_t[:, ::1] track_img,
                       Py_ssize_t cols):

    cdef Py_ssize_t r, c, offset, c_idx
    cdef Py_ssize_t rows = energy_img.shape[0]
    cdef cnp.double_t min_cost = DBL_MAX

    for c in range(cols):
        cumulative_img[0, c] = energy_img[0, c, 0]


    for r in range(1, rows):
        for c in range(cols):
            min_cost = DBL_MAX
            for offset in range(-1, 2):

                c_idx = c + offset
                if (c_idx > cols - 1) or (c_idx < 0) :
                    continue

                if cumulative_img[r-1, c_idx] < min_cost:
                    min_cost = cumulative_img[r-1, c_idx]
                    track_img[r, c] = offset

            #print "min_cost = ", min_cost
            cumulative_img[r,c] = min_cost + energy_img[r, c, 0]

    #print "-------Cumulative Image --------"
    #print np.array(cumulative_img)
    #print "-------Energy Image --------"
    #print np.array(energy_img)

cdef cnp.uint8_t mark_seam(cnp.int8_t[:, ::1] track_img, Py_ssize_t start_index,
                          cnp.uint8_t[:, ::1] seam_map):

    cdef Py_ssize_t rows = track_img.shape[0]
    cdef Py_ssize_t[::1] current_seam_indices = np.zeros(rows, dtype=np.int)
    cdef Py_ssize_t row, col
    cdef cnp.int8_t offset
    cdef Py_ssize_t seams

    current_seam_indices[rows - 1] = start_index
    for row in range(rows - 2, -1, -1):
        col = current_seam_indices[row+1]
        offset = track_img[row, col]
        col = col + offset
        current_seam_indices[row] = col

        if seam_map[row, col]:
            #print "---------- Seam conflict at ", row, col
            return 0

    for row in range(rows):
        col = current_seam_indices[row]
        seam_map[row, col] = 1

    return 1

cdef remove_seam(cnp.double_t[:, :, ::1] img,
                cnp.uint8_t[:, ::1] seam_map, Py_ssize_t cols):

    cdef Py_ssize_t rows = img.shape[0]
    cdef Py_ssize_t channels = img.shape[2]
    cdef Py_ssize_t r, c, ch, shift

    for r in range(rows):
        shift = 0
        for c in range(cols):
            shift += seam_map[r, c]
            for ch in range(channels):
                img[r, c, ch] = img[r, c + shift, ch]

def _seam_carve_v(img, energy_map, iters, border):
    """ Carve vertical seams off an image.

    Carves out vertical seams off an image while using the given energy
    map to decide the importance of each pixel.[1]

    Parameters
    ----------
    img : (M, N) or (M, N, 3) ndarray
        Input image whose vertical seams are to be removed.
    iters : int
        Number of vertical seams are to be removed.
    energy_map : (M, N) ndarray
        The array to decide the importance of each pixel. The higher
        the value corresponding to a pixel, the more the algorithm will try
        to keep it in the image.
    num : int
        Number of seams are to be removed.
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
           http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Avidan07.pdf
    """
    last_row_obj = np.zeros(img.shape[1], dtype=np.float)

    cdef cnp.double_t[::1] last_row = last_row_obj
    cdef Py_ssize_t[::1] sorted_indices
    cdef cnp.uint8_t[:, ::1] seam_map = np.zeros(img.shape[0:2], dtype=np.uint8)
    cdef Py_ssize_t cols = img.shape[1]
    cdef Py_ssize_t rows = img.shape[0]
    cdef Py_ssize_t seams_left = iters
    cdef Py_ssize_t seams_removed
    cdef Py_ssize_t seam_idx

    cdef cnp.double_t[:, :, ::1] image = img
    cdef cnp.int8_t[:, ::1] track_img = np.zeros(img.shape[0:2], dtype=np.int8)
    cdef cnp.double_t[:, ::1] cumulative_img = np.zeros(img.shape[0:2], dtype=np.float)
    cdef cnp.double_t[:, :, ::1] energy_img

    energy_map[:, 0:border] = DBL_MAX
    energy_map[:, cols-border:cols] = DBL_MAX
    energy_map[rows-border:rows, :] = energy_map[rows-2*border:rows-border, :]

    energy_map = np.ascontiguousarray(energy_map[:, :, np.newaxis])
    energy_img = energy_map

    _preprocess_image(energy_img, cumulative_img, track_img, cols)
    last_row[...] = cumulative_img[-1, :]
    sorted_indices = np.argsort(last_row_obj)
    seam_idx = 0

    while seams_left > 0:
        if mark_seam(track_img, sorted_indices[seam_idx], seam_map):
            seams_left -= 1
            cols -= 1
            seam_idx += 1
            continue
        else:
            seam_idx = 0
            remove_seam(image, seam_map, cols)
            remove_seam(energy_img, seam_map, cols)
            seam_map[...] = 0
            _preprocess_image(energy_img, cumulative_img, track_img, cols)
            last_row[:cols] = cumulative_img[-1, :cols]
            sorted_indices = np.argsort(last_row_obj)

    return img[:, 0:cols]
