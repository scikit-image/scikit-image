# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np

# cnp.float64_t[:, :] magnitude
cdef float CellHog(np.ndarray[np.float64_t, ndim=2] magnitude, 
    np.ndarray[np.float64_t, ndim=2] orientation,
    float ori1, float ori2,
    int cx, int cy, int xi, int yi, int sx, int sy):
    """CellHog

    Parameters
    ----------
    magnitude : ndarray
        Coordinate to be clipped.
    orientation : ndarray
        The lower bound.
    ori1 : float
        The higher bound.
    ori2 : float
        The higher bound.
    cx : int
        The higher bound.
    cy : int
        The higher bound.
    xi : int
        The higher bound.
    yi : int
        The higher bound.
    sx : int
        The higher bound.
    sy : int
        The higher bound.

    Returns
    -------
    total : float
        The total HOG value.
    """
    cdef int cx1, cy1

    cdef float total = 0.
    for cy1 in range(-cy/2, cy/2):
        for cx1 in range(-cx/2, cx/2):
            if (yi + cy1 < 0 
                or yi + cy1 >= sy 
                or xi + cx1 < 0
                or xi + cx1 >= sx 
                or orientation[yi + cy1, xi + cx1] >= ori1
                or orientation[yi + cy1, xi + cx1] < ori2): continue

            total += magnitude[yi + cy1, xi + cx1]

    return total

def HogHistograms(np.ndarray[np.float64_t, ndim=2] gx, 
    np.ndarray[np.float64_t, ndim=2] gy, 
    int cx, int cy, #Pixels per cell
    int sx, int sy, #Image size
    int n_cellsx, int n_cellsy, 
    int visualise, int orientations, 
    np.ndarray[np.float64_t, ndim=3] orientation_histogram):
    """HogHistograms

    Parameters
    ----------
    gx : ndarray
        Coordinate to be clipped.
    gy : ndarray
        The lower bound.
    cx : int
        The higher bound.
    cy : int
        The higher bound.
    sx : int
        The higher bound.
    sy : int
        The higher bound.
    n_cellsx : int
        The higher bound.
    n_cellsy : int
        The higher bound.
    visualise : int
        The higher bound.
    orientations : int
        The higher bound.
    orientation_histogram : ndarray
        The histogram to fill.
    """

    cdef np.ndarray[np.float64_t, ndim=2] magnitude = np.hypot(gx, gy)
    cdef np.ndarray[np.float64_t, ndim=2] orientation = (
        np.arctan2(gy, gx) * (180 / np.pi) % 180)
    cdef int i, x, y, o, yi, xi, cy1, cy2, cx1, cx2
    cdef float ori1, ori2

    # compute orientations integral images

    for i in range(orientations):
        # isolate orientations in this range

        ori1 = 180. / orientations * (i + 1)
        ori2 = 180. / orientations * i

        y = cy / 2
        cy2 = cy * n_cellsy
        x = cx / 2
        cx2 = cx * n_cellsx
        yi = 0
        xi = 0

        while y < cy2:
            xi = 0
            x = cx / 2

            while x < cx2:
                orientation_histogram[yi, xi, i] = CellHog(magnitude, 
                    orientation, ori1, ori2, cx, cy, x, y, sx, sy)
                xi += 1
                x += cx

            yi += 1
            y += cy

