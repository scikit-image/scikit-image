# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp

cdef float cell_hog(cnp.float64_t[:, :] magnitude,
        cnp.float64_t[:, :] orientation,
        float orientation_start, float orientation_end,
        int cell_columns, int cell_rows,
        int column_index, int row_index,
        int size_columns, int size_rows):
    """Calculation of the cell's HOG value

    Parameters
    ----------
    magnitude : ndarray
        The gradient magnitudes of the pixels.
    orientation : ndarray
        Lookup table for orientations.
    orientation_start : float
        Orientation range start.
    orientation_end : float
        Orientation range end.
    cell_columns : int
        Pixels per cell (x).
    cell_rows : int
        Pixels per cell (y).
    column_index : int
        Block column index.
    row_index : int
        Block row index.
    size_columns : int
        Number of columns.
    size_rows : int
        Number of rows.

    Returns
    -------
    total : float
        The total HOG value.
    """
    cdef int cell_column, cell_row

    cdef float total = 0.
    for cell_row in range(-cell_rows/2, cell_rows/2):
        for cell_column in range(-cell_columns/2, cell_columns/2):
            if (row_index + cell_row < 0
                or row_index + cell_row >= size_rows
                or column_index + cell_column < 0
                or column_index + cell_column >= size_columns
                or orientation[row_index + cell_row, column_index + cell_column] >= orientation_start
                or orientation[row_index + cell_row, column_index + cell_column] < orientation_end): continue

            total += magnitude[row_index + cell_row, column_index + cell_column]

    return total

def hog_histograms(cnp.float64_t[:, :] gradient_columns,
       cnp.float64_t[:, :] gradient_rows,
       int cell_columns, int cell_rows,
       int size_columns, int size_rows,
       int number_of_cells_columns, int number_of_cells_rows,
       int number_of_orientations,
       cnp.float64_t[:, :, :] orientation_histogram):
    """Extract Histogram of Oriented Gradients (HOG) for a given image.

    Parameters
    ----------
    gradient_columns : ndarray
        First order image gradients (x).
    gradient_rows : ndarray
        First order image gradients (y).
    cell_columns : int
        Pixels per cell (x).
    cell_rows : int
        Pixels per cell (y).
    size_columns : int
        Number of columns.
    size_rows : int
        Number of rows.
    number_of_cells_columns : int
        Number of cells (x).
    number_of_cells_rows : int
        Number of cells (y).
    number_of_orientations : int
        Number of orientation bins.
    orientation_histogram : ndarray
        The histogram to fill.
    """

    cdef cnp.float64_t[:, :] magnitude = np.hypot(gradient_columns, gradient_rows)
    cdef cnp.float64_t[:, :] orientation = np.arctan2(gradient_rows, gradient_columns) * (180 / np.pi) % 180
    cdef int i, x, y, o, yi, xi, cy1, cy2, cx1, cx2
    cdef float orientation_start, orientation_end

    # compute orientations integral images

    for i in range(number_of_orientations):
        # isolate orientations in this range

        orientation_start = 180. / number_of_orientations * (i + 1)
        orientation_end = 180. / number_of_orientations * i

        y = cell_rows / 2
        cy2 = cell_rows * number_of_cells_rows
        x = cell_columns / 2
        cx2 = cell_columns * number_of_cells_columns
        yi = 0
        xi = 0

        while y < cy2:
            xi = 0
            x = cell_columns / 2

            while x < cx2:
                orientation_histogram[yi, xi, i] = cell_hog(magnitude,
                    orientation, orientation_start, orientation_end, cell_columns, cell_rows, x, y, size_columns, size_rows)
                xi += 1
                x += cell_columns

            yi += 1
            y += cell_rows

