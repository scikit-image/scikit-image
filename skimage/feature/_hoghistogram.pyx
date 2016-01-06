# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp

cdef float cell_hog(double[:, ::1] magnitude,
                    double[:, ::1] orientation,
                    float orientation_start, float orientation_end,
                    int cell_columns, int cell_rows,
                    int column_index, int row_index,
                    int size_columns, int size_rows,
                    int range_rows_start, int range_rows_stop,
                    int range_columns_start, int range_columns_stop) nogil:
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
        Pixels per cell (rows).
    cell_rows : int
        Pixels per cell (columns).
    column_index : int
        Block column index.
    row_index : int
        Block row index.
    size_columns : int
        Number of columns.
    size_rows : int
        Number of rows.
    range_rows_start : int
        Start row of cell.
    range_rows_stop : int
        Stop row of cell.
    range_columns_start : int
        Start column of cell.
    range_columns_stop : int
        Stop column of cell

    Returns
    -------
    total : float
        The total HOG value.
    """
    cdef int cell_column, cell_row, cell_row_index, cell_column_index
    cdef float total = 0.

    for cell_row in range(range_rows_start, range_rows_stop):
        cell_row_index = row_index + cell_row
        if (cell_row_index < 0 or cell_row_index >= size_rows):
            continue

        for cell_column in range(range_columns_start, range_columns_stop):
            cell_column_index = column_index + cell_column
            if (cell_column_index < 0 or cell_column_index >= size_columns
                    or orientation[cell_row_index, cell_column_index]
                    >= orientation_start
                    or orientation[cell_row_index, cell_column_index]
                    < orientation_end):
                continue

            total += magnitude[cell_row_index, cell_column_index]

    return total / (cell_rows * cell_columns)

def hog_histograms(double[:, ::1] gradient_columns,
                   double[:, ::1] gradient_rows,
                   int cell_columns, int cell_rows,
                   int size_columns, int size_rows,
                   int number_of_cells_columns, int number_of_cells_rows,
                   int number_of_orientations,
                   cnp.float64_t[:, :, :] orientation_histogram):
    """Extract Histogram of Oriented Gradients (HOG) for a given image.

    Parameters
    ----------
    gradient_columns : ndarray
        First order image gradients (rows).
    gradient_rows : ndarray
        First order image gradients (columns).
    cell_columns : int
        Pixels per cell (rows).
    cell_rows : int
        Pixels per cell (columns).
    size_columns : int
        Number of columns.
    size_rows : int
        Number of rows.
    number_of_cells_columns : int
        Number of cells (rows).
    number_of_cells_rows : int
        Number of cells (columns).
    number_of_orientations : int
        Number of orientation bins.
    orientation_histogram : ndarray
        The histogram array which is modified in place.
    """

    cdef double[:, ::1] magnitude = np.hypot(gradient_columns,
                                                  gradient_rows)
    cdef double[:, ::1] orientation = \
        np.arctan2(gradient_rows, gradient_columns) * (180 / np.pi) % 180
    cdef int i, x, y, o, yi, xi, cc, cr, x0, y0, \
        range_rows_start, range_rows_stop, \
        range_columns_start, range_columns_stop
    cdef float orientation_start, orientation_end, \
        number_of_orientations_per_180

    x0 = cell_columns / 2
    y0 = cell_rows / 2
    cc = cell_rows * number_of_cells_rows
    cr = cell_columns * number_of_cells_columns
    number_of_orientations_per_180 = 180. / number_of_orientations
    range_rows_stop = cell_rows/2
    range_rows_start = -range_rows_stop
    range_columns_stop = cell_columns/2
    range_columns_start = -range_columns_stop

    with nogil:
        # compute orientations integral images
        for i in range(number_of_orientations):
            # isolate orientations in this range
            orientation_start = number_of_orientations_per_180 * (i + 1)
            orientation_end = number_of_orientations_per_180 * i
            x = x0
            y = y0
            yi = 0
            xi = 0

            while y < cc:
                xi = 0
                x = x0

                while x < cr:
                    orientation_histogram[yi, xi, i] = cell_hog(magnitude,
                        orientation, orientation_start, orientation_end,
                        cell_columns, cell_rows, x, y, size_columns, size_rows,
                        range_rows_start, range_rows_stop, range_columns_start, range_columns_stop)
                    xi += 1
                    x += cell_columns

                yi += 1
                y += cell_rows
