#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp


cdef inline double _get_fraction(double from_value, double to_value,
                                 double level):
    if (to_value == from_value):
        return 0
    return ((level - from_value) / (to_value - from_value))


def unpack_unique_verts(list trilist):
    """
    Convert a list of lists of tuples corresponding to triangle vertices
    into a unique vertex list, and a list of triangle faces w/indices
    corresponding to entries of the vertex list.

    """
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t n_tris = len(trilist)
    cdef Py_ssize_t i, j
    cdef dict vert_index = {}
    cdef list vert_list = []
    cdef list face_list = []
    cdef list templist

    # Iterate over triangles
    for i in range(n_tris):
        templist = []

        # Only parse vertices from non-degenerate triangles
        if not ((trilist[i][0] == trilist[i][1]) or
                (trilist[i][0] == trilist[i][2]) or
                (trilist[i][1] == trilist[i][2])):

            # Iterate over vertices within each triangle
            for j in range(3):
                vert = trilist[i][j]

                # Check if a new unique vertex found
                if vert not in vert_index:
                    vert_index[vert] = idx
                    templist.append(idx)
                    vert_list.append(vert)
                    idx += 1
                else:
                    templist.append(vert_index[vert])

            face_list.append(templist)

    return vert_list, face_list


def iterate_and_store_3d(double[:, :, ::1] arr, double level):
    """Iterate across the given array in a marching-cubes fashion,
    looking for volumes with edges that cross 'level'. If such a volume is
    found, appropriate triangulations are added to a growing list of
    faces to be returned by this function.

    """
    if arr.shape[0] < 2 or arr.shape[1] < 2 or arr.shape[2] < 2:
        raise ValueError("Input array must be at least 2x2x2.")

    cdef list face_list = []
    cdef list norm_list = []
    cdef Py_ssize_t n
    cdef bint plus_z
    plus_z = False

    # The plan is to iterate a 2x2x2 cube across the input array. This means
    # the upper-left corner of the cube needs to iterate across a sub-array
    # of size one-less-large in each direction (so we can get away with no
    # bounds checking in Cython). The cube is represented by eight vertices:
    # v1, v2, ..., v8, oriented thus (see Lorensen, Figure 4):
    #
    #           v8 ------ v7
    #          / |       / |        y
    #         /  |      /  |        ^  z
    #       v4 ------ v3   |        | /
    #        |  v5 ----|- v6        |/          (note: NOT right handed!)
    #        |  /      |  /          ----> x
    #        | /       | /
    #       v1 ------ v2
    #
    # We also maintain the current 2D coordinates for v1, and ensure the array
    # is of type 'double' and is C-contiguous (last index varies fastest).

    # Coords start at (0, 0, 0).
    cdef Py_ssize_t[3] coords
    coords[0] = 0
    coords[1] = 0
    coords[2] = 0

    # Calculate the number of iterations we'll need
    cdef Py_ssize_t num_cube_steps = ((arr.shape[0] - 1) *
                                      (arr.shape[1] - 1) *
                                      (arr.shape[2] - 1))

    cdef unsigned char cube_case = 0
    cdef tuple e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12
    cdef double v1, v2, v3, v4, v5, v6, v7, v8
    cdef Py_ssize_t x0, y0, z0, x1, y1, z1
    e5, e6, e7, e8 = (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)

    for n in range(num_cube_steps):
        # There are 255 unique values for `cube_case`. This algorithm follows
        # the Lorensen paper in vertex and edge labeling, however, it should
        # be noted that Lorensen used a left-handed coordinate system while
        # NumPy uses a proper right handed system. Transforming between these
        # coordinate systems was handled in the definitions of the cube
        # vertices v1, v2, ..., v8.
        #
        # Refer to the paper, figure 4, for cube edge designations e1, ... e12

        # Standard Py_ssize_t coordinates for indexing
        x0, y0, z0 = coords[0], coords[1], coords[2]
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

        # We use a right-handed coordinate system, UNlike the paper, but want
        # to index in agreement - the coordinate adjustment takes place here.
        v1 = arr[x0, y0, z0]
        v2 = arr[x1, y0, z0]
        v3 = arr[x1, y1, z0]
        v4 = arr[x0, y1, z0]
        v5 = arr[x0, y0, z1]
        v6 = arr[x1, y0, z1]
        v7 = arr[x1, y1, z1]
        v8 = arr[x0, y1, z1]

        # Unique triangulation cases
        cube_case = 0
        if (v1 > level): cube_case += 1
        if (v2 > level): cube_case += 2
        if (v3 > level): cube_case += 4
        if (v4 > level): cube_case += 8
        if (v5 > level): cube_case += 16
        if (v6 > level): cube_case += 32
        if (v7 > level): cube_case += 64
        if (v8 > level): cube_case += 128

        if (cube_case != 0 and cube_case != 255):
            # Only do anything if there's a plane intersecting the cube.
            # Cases 0 and 255 are entirely below/above the contour.

            if cube_case > 127:
                if ((cube_case != 150) and
                     (cube_case != 170) and
                     (cube_case != 195)):
                    cube_case = 255 - cube_case

            # Calculate cube edges, to become triangulation vertices.
            # If we moved in a convenient direction, save 1/3 of the effort by
            # re-assigning prior results.
            if plus_z:
                # Reassign prior calculated edges
                e1 = e5
                e2 = e6
                e3 = e7
                e4 = e8
            else:
                # Calculate these edges normally
                e1 = x0 + _get_fraction(v1, v2, level), y0, z0
                e2 = x1, y0 + _get_fraction(v2, v3, level), z0
                e3 = x0 + _get_fraction(v4, v3, level), y1, z0
                e4 = x0, y0 + _get_fraction(v1, v4, level), z0

            # These must be calculated at each point unless we implemented a
            # large, growing lookup table for all adjacent values; could save
            # ~30% in terms of runtime at the expense of memory usage and
            # much greater complexity.
            e5 = x0 + _get_fraction(v5, v6, level), y0, z1
            e6 = x1, y0 + _get_fraction(v6, v7, level), z1
            e7 = x0 + _get_fraction(v8, v7, level), y1, z1
            e8 = x0, y0 + _get_fraction(v5, v8, level), z1
            e9 = x0, y0, z0 + _get_fraction(v1, v5, level)
            e10 = x1, y0, z0 + _get_fraction(v2, v6, level)
            e11 = x0, y1, z0 + _get_fraction(v4, v8, level)
            e12 = x1, y1, z0 + _get_fraction(v3, v7, level)


            # Append appropriate triangles to the growing output `face_list`
            _append_tris(face_list, cube_case, e1, e2, e3, e4, e5,
                           e6, e7, e8, e9, e10, e11, e12)

        # Advance the coords indices
        if coords[2] < arr.shape[2] - 2:
            coords[2] += 1
            plus_z = True
        elif coords[1] < arr.shape[1] - 2:
            coords[1] += 1
            coords[2] = 0
            plus_z = False
        else:
            coords[0] += 1
            coords[1] = 0
            coords[2] = 0
            plus_z = False

    return face_list


def _append_tris(list face_list, unsigned char case, tuple e1, tuple e2,
                 tuple e3, tuple e4, tuple e5, tuple e6, tuple e7, tuple e8,
                 tuple e9, tuple e10, tuple e11, tuple e12):
    # Permits recursive use for duplicated planes to conserve code - it's
    # quite long enough as-is.

    if (case == 1):
        # front lower left corner
        face_list.append([e1, e4, e9])
    elif (case == 2):
        # front lower right corner
        face_list.append([e10, e2, e1])
    elif (case == 3):
        # front lower plane
        face_list.append([e2, e4, e9])
        face_list.append([e2, e9, e10])
    elif (case == 4):
        # front upper right corner
        face_list.append([e12, e3, e2])
    elif (case == 5):
        # lower left, upper right corners
        _append_tris(face_list, 1, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 4, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 6):
        # front right plane
        face_list.append([e12, e3, e1])
        face_list.append([e12, e1, e10])
    elif (case == 7):
        # Shelf including v1, v2, v3
        face_list.append([e3, e4, e12])
        face_list.append([e4, e9, e12])
        face_list.append([e12, e9, e10])
    elif (case == 8):
        # front upper left corner
        face_list.append([e3, e11, e4])
    elif (case == 9):
        # front left plane
        face_list.append([e3, e11, e9])
        face_list.append([e3, e9, e1])
    elif (case == 10):
        # upper left, lower right corners
        _append_tris(face_list, 2, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 8, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 11):
        # Shelf including v4, v1, v2
        face_list.append([e3, e11, e2])
        face_list.append([e11, e10, e2])
        face_list.append([e11, e9, e10])
    elif (case == 12):
        # front upper plane
        face_list.append([e11, e4, e12])
        face_list.append([e2, e4, e12])
    elif (case == 13):
        # Shelf including v1, v4, v3
        face_list.append([e11, e9, e12])
        face_list.append([e12, e9, e1])
        face_list.append([e12, e1, e2])
    elif (case == 14):
        # Shelf including v2, v3, v4
        face_list.append([e11, e10, e12])
        face_list.append([e11, e4, e10])
        face_list.append([e4, e1, e10])
    elif (case == 15):
        # Plane parallel to x-axis through middle
        face_list.append([e11, e9, e12])
        face_list.append([e12, e9, e10])
    elif (case == 16):
        # back lower left corner
        face_list.append([e8, e9, e5])
    elif (case == 17):
        # lower left plane
        face_list.append([e4, e1, e8])
        face_list.append([e8, e1, e5])
    elif (case == 18):
        # lower left back, lower right front corners
        _append_tris(face_list, 16, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 2, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 19):
        # Shelf including v1, v2, v5
        face_list.append([e8, e4, e2])
        face_list.append([e8, e2, e10])
        face_list.append([e8, e10, e5])
    elif (case == 20):
        # lower left back, upper right front corners
        _append_tris(face_list, 16, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 4, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 21):
        # lower left plane + upper right front corner, v1, v3, v5
        _append_tris(face_list, 17, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 4, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 22):
        # front right plane + lower left back corner, v2, v3, v5
        _append_tris(face_list, 16, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 6, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 23):
        # Rotated case 14 in the paper
        face_list.append([e3, e10, e8])
        face_list.append([e3, e10, e12])
        face_list.append([e8, e10, e5])
        face_list.append([e3, e4, e8])
    elif (case == 24):
        # upper front left, lower back left corners
        _append_tris(face_list, 16, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 8, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 25):
        # Shelf including v1, v4, v5
        face_list.append([e1, e5, e3])
        face_list.append([e3, e8, e11])
        face_list.append([e3, e5, e8])
    elif (case == 26):
        # Three isolated corners
        _append_tris(face_list, 2, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 8, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 16, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 27):
        # Full corner v1, case 9 in paper: (v1, v2, v4, v5)
        face_list.append([e11, e3, e2])
        face_list.append([e11, e2, e10])
        face_list.append([e10, e11, e8])
        face_list.append([e8, e5, e10])
    elif (case == 28):
        # upper front plane + corner v5
        _append_tris(face_list, 16, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 12, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 29):
        # special case of 11 in the paper: (v1, v3, v4, v5)
        face_list.append([e11, e5, e2])
        face_list.append([e11, e12, e2])
        face_list.append([e11, e5, e8])
        face_list.append([e2, e1, e5])
    elif (case == 30):
        # Shelf (v2, v3, v4) and lower left back corner
        _append_tris(face_list, 14, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 16, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 31):
        # Shelf: (v6, v7, v8) by inversion
        face_list.append([e11, e12, e10])
        face_list.append([e11, e8, e10])
        face_list.append([e8, e10, e5])
    elif (case == 32):
        # lower right back corner
        face_list.append([e6, e5, e10])
    elif (case == 33):
        # lower right back, lower left front corners
        _append_tris(face_list, 1, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 32, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 34):
        # lower right plane
        face_list.append([e1, e2, e5])
        face_list.append([e2, e6, e5])
    elif (case == 35):
        # Shelf: v1, v2, v6
        face_list.append([e4, e2, e6])
        face_list.append([e4, e9, e6])
        face_list.append([e6, e9, e5])
    elif (case == 36):
        # upper right front, lower right back corners
        _append_tris(face_list, 32, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 4, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 37):
        # lower left front, upper right front, lower right back corners
        _append_tris(face_list, 32, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 4, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 1, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 38):
        # Shelf: v2, v3, v6
        face_list.append([e3, e1, e5])
        face_list.append([e3, e5, e12])
        face_list.append([e12, e5, e6])
    elif (case == 39):
        # Full corner v2: (v1, v2, v3, v6)
        face_list.append([e3, e4, e5])
        face_list.append([e4, e9, e5])
        face_list.append([e3, e5, e6])
        face_list.append([e3, e12, e6])
    elif (case == 40):
        # upper left front, lower right back corners
        _append_tris(face_list, 32, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 8, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 41):
        # front left plane, lower right back corner
        _append_tris(face_list, 32, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 9, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 42):
        # lower right plane, upper front left corner
        _append_tris(face_list, 8, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 34, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 43):
        # Rotated case 11 in paper
        face_list.append([e11, e3, e9])
        face_list.append([e3, e9, e6])
        face_list.append([e3, e2, e6])
        face_list.append([e9, e5, e6])
    elif (case == 44):
        # upper front plane, lower right back corner
        _append_tris(face_list, 12, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 32, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 45):
        # Shelf: (v1, v3, v4) + lower right back corner
        _append_tris(face_list, 13, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 32, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 46):
        # Rotated case 14 in paper
        face_list.append([e4, e11, e12])
        face_list.append([e4, e12, e5])
        face_list.append([e12, e5, e6])
        face_list.append([e4, e5, e1])
    elif (case == 47):
        # Shelf: (v5, v8, v7) by inversion
        face_list.append([e11, e9, e12])
        face_list.append([e12, e9, e5])
        face_list.append([e12, e5, e6])
    elif (case == 48):
        # Back lower plane
        face_list.append([e9, e10, e6])
        face_list.append([e9, e6, e8])
    elif (case == 49):
        # Shelf: (v1, v5, v6)
        face_list.append([e4, e8, e6])
        face_list.append([e4, e6, e1])
        face_list.append([e6, e1, e10])
    elif (case == 50):
        # Shelf: (v2, v5, v6)
        face_list.append([e8, e6, e2])
        face_list.append([e8, e2, e1])
        face_list.append([e8, e9, e1])
    elif (case == 51):
        # Plane through middle of cube, parallel to x-z axis
        face_list.append([e4, e8, e2])
        face_list.append([e8, e2, e6])
    elif (case == 52):
        # Back lower plane, and front upper right corner
        _append_tris(face_list, 48, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 4, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 53):
        # Shelf (v1, v5, v6) and front upper right corner
        _append_tris(face_list, 49, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 4, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 54):
        # Rotated case 11 from paper (v2, v3, v5, v6)
        face_list.append([e1, e9, e3])
        face_list.append([e9, e3, e6])
        face_list.append([e9, e8, e6])
        face_list.append([e12, e3, e6])
    elif (case == 55):
        # Shelf: (v4, v8, v7) by inversion
        face_list.append([e4, e8, e6])
        face_list.append([e4, e6, e3])
        face_list.append([e6, e3, e12])
    elif (case == 56):
        # Back lower plane + upper left front corner
        _append_tris(face_list, 48, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 8, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 57):
        # Rotated case 14 from paper (v4, v1, v5, v6)
        face_list.append([e3, e11, e8])
        face_list.append([e3, e8, e10])
        face_list.append([e10, e6, e8])
        face_list.append([e3, e1, e10])
    elif (case == 58):
        # Shelf: (v2, v6, v5) + upper left front corner
        _append_tris(face_list, 50, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 8, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 59):
        # Shelf: (v3, v7, v8) by inversion
        face_list.append([e2, e6, e8])
        face_list.append([e8, e2, e3])
        face_list.append([e8, e3, e11])
    elif (case == 60):
        # AMBIGUOUS CASE: parallel planes (front upper, back lower)
        _append_tris(face_list, 48, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 12, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 61):
        # Upper back plane + lower right front corner by inversion
        _append_tris(face_list, 63, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 2, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 62):
        # Upper back plane + lower left front corner by inversion
        _append_tris(face_list, 63, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 1, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 63):
        # Upper back plane
        face_list.append([e11, e12, e6])
        face_list.append([e11, e8, e6])
    elif (case == 64):
        # Upper right back corner
        face_list.append([e12, e7, e6])
    elif (case == 65):
        # upper right back, lower left front corners
        _append_tris(face_list, 64, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 1, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 66):
        # upper right back, lower right front corners
        _append_tris(face_list, 64, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 2, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 67):
        # lower front plane + upper right back corner
        _append_tris(face_list, 64, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 3, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 68):
        # upper right plane
        face_list.append([e3, e2, e6])
        face_list.append([e3, e7, e6])
    elif (case == 69):
        # Upper right plane, lower left front corner
        _append_tris(face_list, 68, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 1, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 70):
        # Shelf: (v2, v3, v7)
        face_list.append([e1, e3, e7])
        face_list.append([e1, e10, e7])
        face_list.append([e7, e10, e6])
    elif (case == 71):
        # Rotated version of case 11 in paper (v1, v2, v3, v7)
        face_list.append([e10, e7, e4])
        face_list.append([e4, e3, e7])
        face_list.append([e10, e4, e9])
        face_list.append([e7, e10, e6])
    elif (case == 72):
        # upper left front, upper right back corners
        _append_tris(face_list, 64, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 8, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 73):
        # front left plane, upper right back corner
        _append_tris(face_list, 64, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 9, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 74):
        # Three isolated corners, exactly case 7 in paper
        _append_tris(face_list, 64, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 8, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 2, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 75):
        # Shelf: (v1, v2, v4) + upper right back corner
        _append_tris(face_list, 64, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 11, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 76):
        # Shelf: (v4, v3, v7)
        face_list.append([e4, e2, e6])
        face_list.append([e4, e11, e7])
        face_list.append([e4, e7, e6])
    elif (case == 77):
        # Rotated case 14 in paper (v1, v4, v3, v7)
        face_list.append([e11, e9, e1])
        face_list.append([e11, e1, e6])
        face_list.append([e1, e6, e2])
        face_list.append([e11, e6, e7])
    elif (case == 78):
        # Full corner v3: (v2, v3, v4, v7)
        face_list.append([e1, e4, e7])
        face_list.append([e1, e7, e6])
        face_list.append([e4, e11, e7])
        face_list.append([e1, e10, e6])
    elif (case == 79):
        # Shelf: (v6, v5, v8) by inversion
        face_list.append([e9, e11, e10])
        face_list.append([e11, e7, e10])
        face_list.append([e7, e10, e6])
    elif (case == 80):
        # lower left back, upper right back corners (v5, v7)
        _append_tris(face_list, 64, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 16, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 81):
        # lower left plane, upper right back corner
        _append_tris(face_list, 64, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 17, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 82):
        # isolated corners (v2, v5, v7)
        _append_tris(face_list, 64, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 16, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 2, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 83):
        # Shelf: (v1, v2, v5) + upper right back corner
        _append_tris(face_list, 64, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 19, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 84):
        # upper right plane, lower left back corner
        _append_tris(face_list, 16, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 68, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 85):
        # AMBIGUOUS CASE: upper right and lower left parallel planes
        _append_tris(face_list, 17, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 68, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 86):
        # Shelf: (v2, v3, v7) + lower left back corner
        _append_tris(face_list, 70, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 16, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 87):
        # Upper left plane + lower right back corner, by inversion
        _append_tris(face_list, 119, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 32, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 88):
        # Isolated corners v4, v5, v7
        _append_tris(face_list, 64, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 16, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 8, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 89):
        # Shelf: (v1, v4, v5) + isolated corner v7
        _append_tris(face_list, 25, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 64, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 90):
        # Four isolated corners v2, v4, v5, v7
        _append_tris(face_list, 2, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 8, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 16, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 64, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 91):
        # Three isolated corners, v3, v6, v8 by inversion
        _append_tris(face_list, 4, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 32, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 127, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 92):
        # Shelf (v4, v3, v7) + isolated corner v5
        _append_tris(face_list, 76, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 16, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 93):
        # Lower right plane + isolated corner v8 by inversion
        _append_tris(face_list, 127, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 34, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 94):
        # Isolated corners v1, v6, v8 by inversion
        _append_tris(face_list, 1, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 32, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 127, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 95):
        # Isolated corners v6, v8 by inversion
        _append_tris(face_list, 32, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 127, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 96):
        # back right plane
        face_list.append([e7, e12, e5])
        face_list.append([e5, e10, e12])
    elif (case == 97):
        # back right plane + isolated corner v1
        _append_tris(face_list, 96, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 1, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 98):
        # Shelf: (v2, v6, v7)
        face_list.append([e1, e7, e5])
        face_list.append([e7, e1, e12])
        face_list.append([e1, e12, e2])
    elif (case == 99):
        # Rotated case 14 in paper: (v1, v2, v6, v7)
        face_list.append([e9, e2, e7])
        face_list.append([e9, e2, e4])
        face_list.append([e2, e7, e12])
        face_list.append([e7, e9, e5])
    elif (case == 100):
        # Shelf: (v3, v6, v7)
        face_list.append([e3, e7, e5])
        face_list.append([e3, e5, e2])
        face_list.append([e2, e5, e10])
    elif (case == 101):
        # Shelf: (v3, v6, v7) + isolated corner v1
        _append_tris(face_list, 100, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 1, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 102):
        # Plane bisecting left-right halves of cube
        face_list.append([e1, e3, e7])
        face_list.append([e1, e7, e5])
    elif (case == 103):
        # Shelf: (v4, v5, v8) by inversion
        face_list.append([e3, e7, e5])
        face_list.append([e3, e5, e4])
        face_list.append([e4, e5, e9])
    elif (case == 104):
        # Back right plane + isolated corner v4
        _append_tris(face_list, 8, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 96, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 105):
        # AMBIGUOUS CASE: back right and front left planes
        _append_tris(face_list, 96, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 9, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 106):
        # Shelf: (v2, v6, v7) + isolated corner v4
        _append_tris(face_list, 98, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 8, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 107):
        # Back left plane + isolated corner v3 by inversion
        _append_tris(face_list, 4, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 111, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 108):
        # Rotated case 11 from paper: (v4, v3, v7, v6)
        face_list.append([e4, e10, e7])
        face_list.append([e4, e10, e2])
        face_list.append([e4, e11, e7])
        face_list.append([e7, e10, e5])
    elif (case == 109):
        # Back left plane + isolated corner v2 by inversion
        _append_tris(face_list, 111, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 2, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 110):
        # Shelf: (v1, v5, v8) by inversion
        face_list.append([e1, e5, e7])
        face_list.append([e1, e7, e11])
        face_list.append([e1, e11, e4])
    elif (case == 111):
        # Back left plane
        face_list.append([e11, e9, e7])
        face_list.append([e9, e7, e5])
    elif (case == 112):
        # Shelf: (v5, v6, v7)
        face_list.append([e9, e10, e12])
        face_list.append([e9, e12, e7])
        face_list.append([e9, e7, e8])
    elif (case == 113):
        # Exactly case 11 from paper: (v1, v5, v6, v7)
        face_list.append([e1, e8, e12])
        face_list.append([e1, e8, e4])
        face_list.append([e8, e7, e12])
        face_list.append([e12, e1, e10])
    elif (case == 114):
        # Full corner v6: (v2, v6, v7, v5)
        face_list.append([e1, e9, e7])
        face_list.append([e1, e7, e12])
        face_list.append([e1, e12, e2])
        face_list.append([e9, e8, e7])
    elif (case == 115):
        # Shelf: (v3, v4, v8)
        face_list.append([e2, e4, e8])
        face_list.append([e2, e12, e7])
        face_list.append([e2, e8, e7])
    elif (case == 116):
        # Rotated case 14 in paper: (v5, v6, v7, v3)
        face_list.append([e9, e2, e7])
        face_list.append([e9, e2, e10])
        face_list.append([e9, e8, e7])
        face_list.append([e2, e3, e7])
    elif (case == 117):
        # upper left plane + isolated corner v2 by inversion
        _append_tris(face_list, 2, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 119, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 118):
        # Shelf: (v1, v4, v8)
        face_list.append([e1, e3, e7])
        face_list.append([e7, e1, e8])
        face_list.append([e1, e8, e9])
    elif (case == 119):
        # Upper left plane
        face_list.append([e4, e3, e7])
        face_list.append([e4, e8, e7])
    elif (case == 120):
        # Shelf: (v5, v6, v7) + isolated corner v4
        _append_tris(face_list, 112, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 8, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 121):
        # Front right plane + isolated corner v8
        _append_tris(face_list, 6, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 127, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 122):
        # Isolated corners v1, v3, v8
        _append_tris(face_list, 1, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 4, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 127, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 123):
        # Isolated corners v3, v8
        _append_tris(face_list, 4, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 127, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 124):
        # Front lower plane + isolated corner v8
        _append_tris(face_list, 3, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 127, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 125):
        # Isolated corners v2, v8
        _append_tris(face_list, 2, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 127, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 126):
        # Isolated corners v1, v8
        _append_tris(face_list, 1, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 127, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 127):
        # Isolated corner v8
        face_list.append([e11, e7, e8])
    elif (case == 150):
        # AMBIGUOUS CASE: back right and front left planes
        # In these cube_case > 127 cases, the vertices are identical BUT
        # they are connected in the opposite fashion.
        _append_tris(face_list, 6, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 111, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 170):
        # AMBIGUOUS CASE: upper left and lower right planes
        # In these cube_case > 127 cases, the vertices are identical BUT
        # they are connected in the opposite fashion.
        _append_tris(face_list, 119, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 34, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
    elif (case == 195):
        # AMBIGUOUS CASE: back upper and front lower planes
        # In these cube_case > 127 cases, the vertices are identical BUT
        # they are connected in the opposite fashion.
        _append_tris(face_list, 63, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)
        _append_tris(face_list, 3, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                     e11, e12)

    return
