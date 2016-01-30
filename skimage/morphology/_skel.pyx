"""
This is an implementation of the 2D/3D thinning algorithm
of [Lee94]_ of binary images, based on [IAC15]_. 

The original Java code [IAC15]_ carries the following message:

 * This work is an implementation by Ignacio Arganda-Carreras of the
 * 3D thinning algorithm from Lee et al. "Building skeleton models via 3-D 
 * medial surface/axis thinning algorithms. Computer Vision, Graphics, and 
 * Image Processing, 56(6):462–478, 1994." Based on the ITK version from
 * Hanno Homann <a href="http://hdl.handle.net/1926/1292"> http://hdl.handle.net/1926/1292</a>
 * <p>
 *  More information at Skeletonize3D homepage:
 *  http://fiji.sc/Skeletonize3D
 *
 * @version 1.0 11/13/2015 (unique BSD licensed version for scikit-image)
 * @author Ignacio Arganda-Carreras (iargandacarreras at gmail.com)

Porting to Cython was done by Evgeni Burovski (evgeny.burovskiy@gmail.com).

References
----------

.. [Lee94] Lee et al, Building skeleton models via 3-D medial surface/axis
           thinning algorithms. Computer Vision, Graphics, and Image Processing,
           56(6):462–478, 1994

.. [IAC15] Ignacio Arganda-Carreras, 2015. Skeletonize3D plugin for ImageJ(C).
           http://fiji.sc/Skeletonize3D

"""
from __future__ import division, print_function, absolute_import

from libc.string cimport memcpy

import numpy as np
from numpy cimport npy_intp, npy_uint8
cimport cython

ctypedef npy_uint8 pixel_type


@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_thin_image(pixel_type[:, :, ::1] img not None):
    """Compute a thin image.

    Loop through the image multiple times, removing "simple" points, i.e.
    those point which can be removed without changing local connectivity in the
    3x3x3 neighborhood of a point.

    This routine implements the two-pass algorthim of [Lee94]_. Namely,
    for each of the six border types (positive and negative x-, y- and z-),
    the algorithm first collects all possibly deletable points, and then
    performs a sequential rechecking.

    The input, `img`, is assumed to be a 3D binary image in the
    (p, r, c) format [i.e., C ordered array], filled by zeros (background) and
    ones. Furthermore, `img` is assumed to be padded by zeros from all
    directions --- this way the zero boundary conditions are authomatic
    and there is need to guard against out-of-bounds access.

    """
    cdef:
        int unchanged_borders = 0, curr_border, num_borders
        int borders[6]
        npy_intp p, r, c
        bint no_change
        list simple_border_points
        pixel_type neighb[27]
    borders[:] = [4, 3, 2, 1, 5, 6]

    # no need to worry about the z direction if the original image is 2D.
    if img.shape[0] == 3:
        num_borders = 4
    else:
        num_borders = 6

    # loop through the image several times until there is no change for all
    # the six border types
    while unchanged_borders < num_borders:
        unchanged_borders = 0
        for j in range(num_borders):
            curr_border = borders[j]

            simple_border_points = _loop_through(img, curr_border)
           ## print(curr_border, " : ", simple_border_points, '\n')

            # sequential re-checking to preserve connectivity when deleting
            # in a parallel way
            no_change = True
            for pt in simple_border_points:
                p, r, c = pt
                get_neighborhood(img, p, r, c, neighb)
                if is_simple_point(neighb):
                    img[p, r, c] = 0
                    no_change = False
                else:
                    pass
            ##        print(" *** ", pt, " is not simple.")

            if no_change:
                unchanged_borders += 1
            simple_border_points = []

    return np.asarray(img)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list _loop_through(pixel_type[:, :, ::1] img,
                        int curr_border):
    """Inner loop of compute_thin_image.

    The algorithm of [Lee94]_ proceeds in two steps: (1) six directions are
    checked for simple border points to remove, and (2) these candidates are
    sequentially rechecked, see Sec 3 of [Lee94]_ for rationale and discussion.

    This routine implements the first step above: it loops over the image
    for a given direction and assembles candidates for removal.

    """
    # This routine looks like it could be nogil, but actually it cannot be,
    # because of `simple_border_points` being a python list which is being
    # mutated.
    cdef:
        list simple_border_points = []
        pixel_type neighborhood[27]
        npy_intp p, r, c
        bint is_border_pt

        # rebind global names to avoid lookup. Both tables are filled in
        # at import time.
        int[::1] Euler_LUT = LUT
        cdef int[:, ::1] neighb_idx = NEIGHB_IDX

    # loop through the image
    # NB: each loop is from 1 to size-1: img is padded from all sides 
    for p in range(1, img.shape[0] - 1):
        for r in range(1, img.shape[1] - 1):
            for c in range(1, img.shape[2] - 1):

                # check if pixel is foreground
                if img[p, r, c] != 1:
                    continue

                is_border_pt = (curr_border == 1 and img[p, r, c-1] <= 0 or  #N
                                curr_border == 2 and img[p, r, c+1] <= 0 or  #S
                                curr_border == 3 and img[p, r+1, c] <= 0 or  #E
                                curr_border == 4 and img[p, r-1, c] <= 0 or  #W
                                curr_border == 5 and img[p+1, r, c] <= 0 or  #U
                                curr_border == 6 and img[p-1, r, c] <= 0)    #B
                if not is_border_pt:
                    # current point is not deletable
                    continue

                get_neighborhood(img, p, r, c, neighborhood)

                # check if (p, r, c) is an endpoint (then it's not deletable.)
                if is_endpoint(neighborhood):
                    continue

                # check if point is Euler invariant (condition 1 in [Lee94]_):
                # if it is not, it's not deletable.
                if not is_Euler_invariant(neighborhood, Euler_LUT, neighb_idx):
                    continue

                # check if point is simple (i.e., deletion does not
                # change connectivity in the 3x3x3 neighborhood)
                # this are conditions 2 and 3 in [Lee94]_
                if not is_simple_point(neighborhood):
                    continue

                # ok, add (p, r, c) to the list of simple border points
                simple_border_points.append((p, r, c))
    return simple_border_points


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void get_neighborhood(pixel_type[:, :, ::1] img,
                           npy_intp p, npy_intp r, npy_intp c,
                           pixel_type neighborhood[]):
    """Get the neighborhood of a pixel.

    Assume zero boundary conditions. 
    Image is already padded, so no out-of-bounds checking.

    For the numbering of points see Fig. 1a. of [Lee94]_, where the numbers
    do *not* include the center point itself. OTOH, this numbering below
    includes it as number 13. The latter is consistent with [IAC15]_.
    """
    neighborhood[0] = img[p-1, r-1, c-1]
    neighborhood[1] = img[p-1, r,   c-1]
    neighborhood[2] = img[p-1, r+1, c-1]

    neighborhood[ 3] = img[p-1, r-1, c]
    neighborhood[ 4] = img[p-1, r,   c]
    neighborhood[ 5] = img[p-1, r+1, c]

    neighborhood[ 6] = img[p-1, r-1, c+1]
    neighborhood[ 7] = img[p-1, r,   c+1]
    neighborhood[ 8] = img[p-1, r+1, c+1]

    neighborhood[ 9] = img[p, r-1, c-1]
    neighborhood[10] = img[p, r,   c-1]
    neighborhood[11] = img[p, r+1, c-1]

    neighborhood[12] = img[p, r-1, c]
    neighborhood[13] = img[p, r,   c]
    neighborhood[14] = img[p, r+1, c]

    neighborhood[15] = img[p, r-1, c+1]
    neighborhood[16] = img[p, r,   c+1]
    neighborhood[17] = img[p, r+1, c+1]

    neighborhood[18] = img[p+1, r-1, c-1]
    neighborhood[19] = img[p+1, r,   c-1]
    neighborhood[20] = img[p+1, r+1, c-1]

    neighborhood[21] = img[p+1, r-1, c]
    neighborhood[22] = img[p+1, r,   c]
    neighborhood[23] = img[p+1, r+1, c]

    neighborhood[24] = img[p+1, r-1, c+1]
    neighborhood[25] = img[p+1, r,   c+1]
    neighborhood[26] = img[p+1, r+1, c+1]


###### look-up tables
def fill_Euler_LUT():
    """ Look-up table for preserving Euler characteristic.

    This is column $\delta G_{26}$ of Table 2 of [Lee94]_.
    """
    LUT = np.zeros(256, dtype=np.intc)

    LUT[1]  =  1
    LUT[3]  = -1
    LUT[5]  = -1
    LUT[7]  =  1
    LUT[9]  = -3
    LUT[11] = -1
    LUT[13] = -1
    LUT[15] =  1
    LUT[17] = -1
    LUT[19] =  1
    LUT[21] =  1
    LUT[23] = -1
    LUT[25] =  3
    LUT[27] =  1
    LUT[29] =  1
    LUT[31] = -1
    LUT[33] = -3
    LUT[35] = -1
    LUT[37] =  3
    LUT[39] =  1
    LUT[41] =  1
    LUT[43] = -1
    LUT[45] =  3
    LUT[47] =  1
    LUT[49] = -1
    LUT[51] =  1

    LUT[53] =  1
    LUT[55] = -1
    LUT[57] =  3
    LUT[59] =  1
    LUT[61] =  1
    LUT[63] = -1
    LUT[65] = -3
    LUT[67] =  3
    LUT[69] = -1
    LUT[71] =  1
    LUT[73] =  1
    LUT[75] =  3
    LUT[77] = -1
    LUT[79] =  1
    LUT[81] = -1
    LUT[83] =  1
    LUT[85] =  1
    LUT[87] = -1
    LUT[89] =  3
    LUT[91] =  1
    LUT[93] =  1
    LUT[95] = -1
    LUT[97] =  1
    LUT[99] =  3
    LUT[101] =  3
    LUT[103] =  1

    LUT[105] =  5
    LUT[107] =  3
    LUT[109] =  3
    LUT[111] =  1
    LUT[113] = -1
    LUT[115] =  1
    LUT[117] =  1
    LUT[119] = -1
    LUT[121] =  3
    LUT[123] =  1
    LUT[125] =  1
    LUT[127] = -1
    LUT[129] = -7
    LUT[131] = -1
    LUT[133] = -1
    LUT[135] =  1
    LUT[137] = -3
    LUT[139] = -1
    LUT[141] = -1
    LUT[143] =  1
    LUT[145] = -1
    LUT[147] =  1
    LUT[149] =  1
    LUT[151] = -1
    LUT[153] =  3
    LUT[155] =  1

    LUT[157] =  1
    LUT[159] = -1
    LUT[161] = -3
    LUT[163] = -1
    LUT[165] =  3
    LUT[167] =  1
    LUT[169] =  1
    LUT[171] = -1
    LUT[173] =  3
    LUT[175] =  1
    LUT[177] = -1
    LUT[179] =  1
    LUT[181] =  1
    LUT[183] = -1
    LUT[185] =  3
    LUT[187] =  1
    LUT[189] =  1
    LUT[191] = -1
    LUT[193] = -3
    LUT[195] =  3
    LUT[197] = -1
    LUT[199] =  1
    LUT[201] =  1
    LUT[203] =  3
    LUT[205] = -1
    LUT[207] =  1

    LUT[209] = -1
    LUT[211] =  1
    LUT[213] =  1
    LUT[215] = -1
    LUT[217] =  3
    LUT[219] =  1
    LUT[221] =  1
    LUT[223] = -1
    LUT[225] =  1
    LUT[227] =  3
    LUT[229] =  3
    LUT[231] =  1
    LUT[233] =  5
    LUT[235] =  3
    LUT[237] =  3
    LUT[239] =  1
    LUT[241] = -1
    LUT[243] =  1
    LUT[245] =  1
    LUT[247] = -1
    LUT[249] =  3
    LUT[251] =  1
    LUT[253] =  1
    LUT[255] = -1
    return LUT
cdef int[::1] LUT = fill_Euler_LUT()


# Fill the look-up table for indexing octants for computing the Euler
# characteristic. See index_octants and is_Euler_invariant routines below.
cdef int[:, ::1] NEIGHB_IDX = np.array([[2, 1, 11, 10, 5, 4, 14],      # NEB
                                        [0, 9, 3, 12, 1, 10, 4],       # NWB
                                        [8, 7, 17, 16, 5, 4, 14],      # SEB
                                        [6, 15, 7, 16, 3, 12, 4],      # SWB
                                        [20, 23, 19, 22, 11, 14, 10],  # NEU
                                        [18, 21, 9, 12, 19, 22, 10],   # NWU
                                        [26, 23, 17, 14, 25, 22, 16],  # SEU
                                        [24, 25, 15, 16, 21, 22, 12],  # SWU
                              ], dtype=np.intc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int index_octants(int octant,
                       pixel_type neighbors[],
                       int[:, ::1] neib_idx):
    cdef int n = 1, j, idx
    for j in range(7):
        idx = neib_idx[octant, j]
        if neighbors[idx] == 1:
            n |= 1 << (7 - j)    # XXX hardcode powers?
    return n


cdef inline bint is_endpoint(pixel_type neighbors[]):
    """An endpoint has exactly one neighbor in the 26-neighborhood.
    """
    # The center pixel is counted, thus r.h.s. is 2
    cdef int s = 0, j
    for j in range(27):
        s += neighbors[j]
    return s == 2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint is_Euler_invariant(pixel_type neighbors[],
                             int[::1] lut,
                             int[:, ::1] neighb_idx):
    """Check if a point is Euler invariant.

    Calculate Euler characteristc for each octant and sum up.

    Parameters
    ----------
    neighbors
        neighbors of a point
    lut
        The look-up table for preserving the Euler characteristic.
    neighb_idx
        The look-up table for indexing octants.

    Returns
    -------
    bool (C bool, that is)

    """
    cdef int octant, n, euler_char = 0
    for octant in range(8):
        n = index_octants(octant, neighbors, neighb_idx)
        euler_char += lut[n]
    return euler_char == 0


cdef bint is_simple_point(pixel_type neighbors[]):
    """Check is a point is a Simple Point.

    This method is named "N(v)_labeling" in [Lee94]_.
    Outputs the number of connected objects in a neighborhood of a point
    after this point would have been removed.

    Parameters
    ----------
    neighbors : uint8 C array, shape(27,)
        neighbors of the point

    Returns
    -------
    bool
        Whether the point is simple or not.

    """
    # copy neighbors for labeling
    # ignore center pixel (i=13) when counting (see [Lee94]_)
    cdef pixel_type cube[26]
    memcpy(cube, neighbors, 13*sizeof(pixel_type))
    memcpy(cube+13, neighbors+14, 13*sizeof(pixel_type))

    # set initial label
    cdef int label = 2, i

    # for all point in the neighborhood
    for i in range(26):
        if cube[i] == 1:
            # voxel has not been labeled yet
            # start recursion with any octant that contains the point i
            if i in (0, 1, 3, 4, 9, 10, 12):
                octree_labeling(1, label, cube)
            elif i in (2, 5, 11, 13):
                octree_labeling(2, label, cube)
            elif i in (6, 7, 14, 15):
                octree_labeling(3, label, cube)
            elif i in (8, 16):
                octree_labeling(4, label, cube)
            elif i in (17, 18, 20, 21):
                octree_labeling(5, label, cube)
            elif i in (19, 22):
                octree_labeling(6, label, cube)
            elif i in (23, 24):
                octree_labeling(7, label, cube)
            elif i == 25:
                octree_labeling(8, label, cube)
            else:
                raise ValueError("Never be here. i = %s" % i)
            label += 1
            if label - 2 >= 2:
                return False
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void octree_labeling(int octant, int label, pixel_type cube[]):
    """This is a recursive method that calculates the number of connected
    components in the 3D neighborhood after the center pixel would
    have been removed.

    See Figs. 6 and 7 of [Lee94]_ for the values of indices.

    Parameters
    ----------
    octant : int
        octant index
    label : int 
        the current label of the center point
    cube : uint8 C array, shape(26,)
        local neighborhood of the point

    """
    # check if there are points in the octant with value 1
    if octant == 1:
        # set points in this octant to current label
        # and recursive labeling of adjacent octants
        if cube[0] == 1:
            cube[0] = label
        if cube[1] == 1:
            cube[1] = label
            octree_labeling(2, label, cube)
        if cube[3] == 1:
            cube[3] = label
            octree_labeling(3, label, cube)
        if cube[4] == 1:
            cube[4] = label
            octree_labeling(2, label, cube)
            octree_labeling(3, label, cube)
            octree_labeling(4, label, cube)
        if cube[9] == 1:
            cube[9] = label
            octree_labeling(5, label, cube)
        if cube[10] == 1:
            cube[10] = label
            octree_labeling(2, label, cube)
            octree_labeling(5, label, cube)
            octree_labeling(6, label, cube)
        if cube[12] == 1:
            cube[12] = label
            octree_labeling(3, label, cube)
            octree_labeling(5, label, cube)
            octree_labeling(7, label, cube)

    if octant == 2:
        if cube[1] == 1:
            cube[1] = label
            octree_labeling(1, label, cube)
        if cube[4] == 1:
              cube[4] = label
              octree_labeling(1, label, cube)
              octree_labeling(3, label, cube)
              octree_labeling(4, label, cube)
        if cube[10] == 1:
              cube[10] = label
              octree_labeling(1, label, cube)
              octree_labeling(5, label, cube)
              octree_labeling(6, label, cube)
        if cube[2] == 1:
              cube[2] = label
        if cube[5] == 1:
              cube[5] = label
              octree_labeling(4, label, cube)
        if cube[11] == 1:
              cube[11] = label
              octree_labeling(6, label, cube)
        if cube[13] == 1:
              cube[13] = label
              octree_labeling(4, label, cube)
              octree_labeling(6, label, cube)
              octree_labeling(8, label, cube)

    if octant ==3:
        if cube[3] == 1:
              cube[3] = label
              octree_labeling(1, label, cube)
        if cube[4] == 1:
              cube[4] = label
              octree_labeling(1, label, cube)
              octree_labeling(2, label, cube)
              octree_labeling(4, label, cube)
        if cube[12] == 1:
              cube[12] = label
              octree_labeling(1, label, cube)
              octree_labeling(5, label, cube)
              octree_labeling(7, label, cube)
        if cube[6] == 1:
              cube[6] = label
        if cube[7] == 1:
              cube[7] = label
              octree_labeling(4, label, cube)
        if cube[14] == 1:
              cube[14] = label
              octree_labeling(7, label, cube)
        if cube[15] == 1:
              cube[15] = label
              octree_labeling(4, label, cube)
              octree_labeling(7, label, cube)
              octree_labeling(8, label, cube)

    if octant == 4:
        if cube[4] == 1:
              cube[4] = label
              octree_labeling(1, label, cube)
              octree_labeling(2, label, cube)
              octree_labeling(3, label, cube)
        if cube[5] == 1:
              cube[5] = label
              octree_labeling(2, label, cube)
        if cube[13] == 1:
              cube[13] = label
              octree_labeling(2, label, cube)
              octree_labeling(6, label, cube)
              octree_labeling(8, label, cube)
        if cube[7] == 1:
              cube[7] = label
              octree_labeling(3, label, cube)
        if cube[15] == 1:
              cube[15] = label
              octree_labeling(3, label, cube)
              octree_labeling(7, label, cube)
              octree_labeling(8, label, cube)
        if cube[8] == 1:
              cube[8] = label
        if cube[16] == 1:
              cube[16] = label
              octree_labeling(8, label, cube)

    if octant == 5:
        if cube[9] == 1:
              cube[9] = label
              octree_labeling(1, label, cube)
        if cube[10] == 1:
              cube[10] = label
              octree_labeling(1, label, cube)
              octree_labeling(2, label, cube)
              octree_labeling(6, label, cube)
        if cube[12] == 1:
              cube[12] = label
              octree_labeling(1, label, cube)
              octree_labeling(3, label, cube)
              octree_labeling(7, label, cube)
        if cube[17] == 1:
              cube[17] = label
        if cube[18] == 1:
              cube[18] = label
              octree_labeling(6, label, cube)
        if cube[20] == 1:
              cube[20] = label
              octree_labeling(7, label, cube)
        if cube[21] == 1:
              cube[21] = label
              octree_labeling(6, label, cube)
              octree_labeling(7, label, cube)
              octree_labeling(8, label, cube)

    if octant == 6:
        if cube[10] == 1:
              cube[10] = label
              octree_labeling(1, label, cube)
              octree_labeling(2, label, cube)
              octree_labeling(5, label, cube)
        if cube[11] == 1:
              cube[11] = label
              octree_labeling(2, label, cube)
        if cube[13] == 1:
              cube[13] = label
              octree_labeling(2, label, cube)
              octree_labeling(4, label, cube)
              octree_labeling(8, label, cube)
        if cube[18] == 1:
              cube[18] = label
              octree_labeling(5, label, cube)
        if cube[21] == 1:
              cube[21] = label
              octree_labeling(5, label, cube)
              octree_labeling(7, label, cube)
              octree_labeling(8, label, cube)
        if cube[19] == 1:
              cube[19] = label
        if cube[22] == 1:
              cube[22] = label
              octree_labeling(8, label, cube)

    if octant == 7:
        if cube[12] == 1:
              cube[12] = label
              octree_labeling(1, label, cube)
              octree_labeling(3, label, cube)
              octree_labeling(5, label, cube)
        if cube[14] == 1:
              cube[14] = label
              octree_labeling(3, label, cube)
        if cube[15] == 1:
              cube[15] = label
              octree_labeling(3, label, cube)
              octree_labeling(4, label, cube)
              octree_labeling(8, label, cube)
        if cube[20] == 1:
              cube[20] = label
              octree_labeling(5, label, cube)
        if cube[21] == 1:
              cube[21] = label
              octree_labeling(5, label, cube)
              octree_labeling(6, label, cube)
              octree_labeling(8, label, cube)
        if cube[23] == 1:
              cube[23] = label
        if cube[24] == 1:
              cube[24] = label
              octree_labeling(8, label, cube)

    if octant == 8:
        if cube[13] == 1:
              cube[13] = label
              octree_labeling(2, label, cube)
              octree_labeling(4, label, cube)
              octree_labeling(6, label, cube)
        if cube[15] == 1:
              cube[15] = label
              octree_labeling(3, label, cube)
              octree_labeling(4, label, cube)
              octree_labeling(7, label, cube)
        if cube[16] == 1:
              cube[16] = label
              octree_labeling(4, label, cube)
        if cube[21] == 1:
              cube[21] = label
              octree_labeling(5, label, cube)
              octree_labeling(6, label, cube)
              octree_labeling(7, label, cube)
        if cube[22] == 1:
              cube[22] = label
              octree_labeling(6, label, cube)
        if cube[24] == 1:
              cube[24] = label
              octree_labeling(7, label, cube)
        if cube[25] == 1:
              cube[25] = label
