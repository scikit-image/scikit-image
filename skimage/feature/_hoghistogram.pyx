# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cmath, math
cimport numpy as np
import numpy as np
from scipy import pi, arctan2, cos, sin

cdef float CellHog(np.ndarray[np.float64_t, ndim=2] magnitude, 
    np.ndarray[np.float64_t, ndim=2] orientation,
    float ori1, float ori2,
    int cx, int cy, int xi, int yi, int sx, int sy):
    cdef int cx1, cy1

    cdef float total = 0.
    for cy1 in range(-cy/2, cy/2):
        for cx1 in range(-cx/2, cx/2):
            if yi + cy1 < 0: continue
            if yi + cy1 >= sy: continue
            if xi + cx1 < 0: continue
            if xi + cx1 >= sx: continue
            if orientation[yi + cy1, xi + cx1] >= ori1: continue
            if orientation[yi + cy1, xi + cx1] < ori2: continue

            total += magnitude[yi + cy1, xi + cx1]

    return total

def HogHistograms(np.ndarray[np.float64_t, ndim=2] gx, \
    np.ndarray[np.float64_t, ndim=2] gy, 
    int cx, int cy, #Pixels per cell
    int sx, int sy, #Image size
    int n_cellsx, int n_cellsy, 
    int visualise, int orientations, 
    np.ndarray[np.float64_t, ndim=3] orientation_histogram):

    cdef np.ndarray[np.float64_t, ndim=2] magnitude = np.sqrt(gx**2 + gy**2)
    cdef np.ndarray[np.float64_t, ndim=2] orientation = arctan2(gy, gx) * (180 / pi) % 180
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
                orientation_histogram[yi, xi, i] = CellHog(magnitude, orientation, ori1, ori2, cx, cy, x, y, sx, sy)
                xi += 1
                x += cx

            yi += 1
            y += cy

