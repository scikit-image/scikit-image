#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np


cdef inline double _get_fraction(double from_value, double to_value,
                                 double level):
    if (to_value == from_value):
        return 0
    return ((level - from_value) / (to_value - from_value))


def iterate_and_store(double[:, :] array,
                      double level, Py_ssize_t vertex_connect_high):
    """Iterate across the given array in a marching-squares fashion,
    looking for segments that cross 'level'. If such a segment is
    found, its coordinates are added to a growing list of segments,
    which is returned by the function.  if vertex_connect_high is
    nonzero, high-values pixels are considered to be face+vertex
    connected into objects; otherwise low-valued pixels are.

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

        ul = array[r0, c0]
        ur = array[r0, c1]
        ll = array[r1, c0]
        lr = array[r1, c1]

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
