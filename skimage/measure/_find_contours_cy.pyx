#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np

ctypedef np.uint8_t DTYPE_BOOL_t
cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)
    double NAN "NPY_NAN"

cdef inline double _get_fraction(double from_value, double to_value,
                                 double level):
    if (to_value == from_value):
        return 0
    return ((level - from_value) / (to_value - from_value))


def iterate_and_store(double[:, :] array,
                      double level, Py_ssize_t vertex_connect_high,
                      np.ndarray[DTYPE_BOOL_t, cast=True, ndim=2] mask):
    """Iterate across the given array in a marching-squares fashion,
    looking for segments that cross 'level'. If such a segment is
    found, its coordinates are added to a growing list of segments,
    which is returned by the function.  if vertex_connect_high is
    nonzero, high-values pixels are considered to be face+vertex
    connected into objects; otherwise low-valued pixels are.

    Positions where the boolean array mask is False are considered
    as not containing data.
    """
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("Input array must be at least 2x2.")

    cdef list arc_list = []
    cdef Py_ssize_t n

    # The plan is to iterate a 2x2 square across the input array. This means
    # that the upper-left corner of the square needs to iterate across a
    # sub-array that's one-less-large in each direction (so that the square
    # never steps out of bounds). The square is represented by four pointers:
    # ul, ur, ll, and lr (for 'upper left', etc.). We also maintain the current
    # 2D coordinates for the position of the upper-left pointer. Note that we
    # ensured that the array is of type 'double' and is C-contiguous (last
    # index varies the fastest).

    # Current coords start at 0,0.
    cdef Py_ssize_t[2] coords
    coords[0] = 0
    coords[1] = 0

    # Calculate the number of iterations we'll need
    cdef Py_ssize_t num_square_steps = (array.shape[0] - 1) \
                                        * (array.shape[1] - 1)

    cdef unsigned char square_case = 0
    cdef unsigned char nan_count = 0
    cdef tuple top, bottom, left, right
    cdef double ul, ur, ll, lr
    cdef Py_ssize_t r0, r1, c0, c1

    for n in range(num_square_steps):
        # There are sixteen different possible square types, diagramed below.
        # A + indicates that the vertex is above the contour value, and a -
        # indicates that the vertex is below or equal to the contour value.
        # The vertices of each square are:
        # ul ur
        # ll lr
        # and can be treated as a binary value with the bits in that order. Thus
        # each square case can be numbered:
        #  0--   1+-   2-+   3++   4--   5+-   6-+   7++
        #   --    --    --    --    +-    +-    +-    +-
        #
        #  8--   9+-  10-+  11++  12--  13+-  14-+  15++
        #   -+    -+    -+    -+    ++    ++    ++    ++
        #
        # The position of the line segment that cuts through (or
        # doesn't, in case 0 and 15) each square is clear, except in
        # cases 6 and 9. In this case, where the segments are placed
        # is determined by vertex_connect_high.  If
        # vertex_connect_high is false, then lines like \\ are drawn
        # through square 6, and lines like // are drawn through square
        # 9.  Otherwise, the situation is reversed.
        # Finally, recall that we draw the lines so that (moving from tail to
        # head) the lower-valued pixels are on the left of the line. So, for
        # example, case 1 entails a line slanting from the middle of the top of
        # the square to the middle of the left side of the square.

        r0, c0 = coords[0], coords[1]
        r1, c1 = r0 + 1, c0 + 1

        # Overwrite with NaN where mask is false
        ul = array[r0, c0] if mask[r0, c0] else NAN
        ur = array[r0, c1] if mask[r0, c1] else NAN
        ll = array[r1, c0] if mask[r1, c0] else NAN
        lr = array[r1, c1] if mask[r1, c1] else NAN

        # now in advance the coords indices
        if coords[1] < array.shape[1] - 2:
            coords[1] += 1
        else:
            coords[0] += 1
            coords[1] = 0

        square_case = 0
        if (ul > level): square_case += 1
        if (ur > level): square_case += 2
        if (ll > level): square_case += 4
        if (lr > level): square_case += 8

        # We need to handle missing data.
        # (This could either be in the form of NaNs present in the input array,
        # or could be the result of masking.)
        # Start by counting the number of missing data values.
        nan_count = 0
        if npy_isnan(ul): nan_count += 1
        if npy_isnan(ur): nan_count += 1
        if npy_isnan(ll): nan_count += 1
        if npy_isnan(lr): nan_count += 1

        if nan_count > 1:
            # If a square has 2 or more missing values, we cannot correctly
            # infer the presence of any contour line segments within it;
            # so just move to the next square.
            continue
        elif nan_count == 1:
            # There is (up to symmetry) one square arrangement containing a
            # missing value in which we can unambiguously draw an isoline
            # segment:
            # The arrangement +-
            #                 -x (where x denotes a missing value)
            # should have the same contour line as case 1.
            # After symmetry, there are 8 cases, which are enumerated here.

            # If we match any of them, we adjust square_case to look like
            # case 1 (or the symmetric equivalent) and fall through.
            # Note that NaN values always read as low, since NaN > level
            # is always false; so for +-+x arrangements we don't have to
            # adjust square_case:
            if square_case == 1 and npy_isnan(lr):
                pass
            elif square_case == 2 and npy_isnan(ll):
                pass
            elif square_case == 4 and npy_isnan(ur):
                pass
            elif square_case == 8 and npy_isnan(ul):
                pass

            # For -+-x arrangements, we adjust square_case to mark the NaN
            # as high, not low:
            elif square_case == 6 and npy_isnan(lr):
                square_case = 14
            elif square_case == 9 and npy_isnan(ll):
                square_case = 13
            elif square_case == 9 and npy_isnan(ur):
                square_case = 11
            elif square_case == 6 and npy_isnan(ul):
                square_case = 7

            # If we don't match any, we don't add any contour in this square.
            else:
                continue

        if (square_case != 0 and square_case != 15):
            # only do anything if there's a line passing through the
            # square. Cases 0 and 15 are entirely below/above the contour.

            top = r0, c0 + _get_fraction(ul, ur, level)
            bottom = r1, c0 + _get_fraction(ll, lr, level)
            left = r0 + _get_fraction(ul, ll, level), c0
            right = r0 + _get_fraction(ur, lr, level), c1

            if (square_case == 1):
                # top to left
                arc_list.append(top)
                arc_list.append(left)
            elif (square_case == 2):
                # right to top
                arc_list.append(right)
                arc_list.append(top)
            elif (square_case == 3):
                # right to left
                arc_list.append(right)
                arc_list.append(left)
            elif (square_case == 4):
                # left to bottom
                arc_list.append(left)
                arc_list.append(bottom)
            elif (square_case == 5):
                # top to bottom
                arc_list.append(top)
                arc_list.append(bottom)
            elif (square_case == 6):
                if vertex_connect_high:
                    arc_list.append(left)
                    arc_list.append(top)

                    arc_list.append(right)
                    arc_list.append(bottom)
                else:
                    arc_list.append(right)
                    arc_list.append(top)
                    arc_list.append(left)
                    arc_list.append(bottom)
            elif (square_case == 7):
                # right to bottom
                arc_list.append(right)
                arc_list.append(bottom)
            elif (square_case == 8):
                # bottom to right
                arc_list.append(bottom)
                arc_list.append(right)
            elif (square_case == 9):
                if vertex_connect_high:
                    arc_list.append(top)
                    arc_list.append(right)

                    arc_list.append(bottom)
                    arc_list.append(left)
                else:
                    arc_list.append(top)
                    arc_list.append(left)

                    arc_list.append(bottom)
                    arc_list.append(right)
            elif (square_case == 10):
                # bottom to top
                arc_list.append(bottom)
                arc_list.append(top)
            elif (square_case == 11):
                # bottom to left
                arc_list.append(bottom)
                arc_list.append(left)
            elif (square_case == 12):
                # lef to right
                arc_list.append(left)
                arc_list.append(right)
            elif (square_case == 13):
                # top to right
                arc_list.append(top)
                arc_list.append(right)
            elif (square_case == 14):
                # left to top
                arc_list.append(left)
                arc_list.append(top)

    return arc_list
