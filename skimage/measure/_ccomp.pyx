#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
import warnings

cimport numpy as cnp

"""
See also:

  Christophe Fiorio and Jens Gustedt,
  "Two linear time Union-Find strategies for image processing",
  Theoretical Computer Science 154 (1996), pp. 165-181.

  Kensheng Wu, Ekow Otoo and Arie Shoshani,
  "Optimizing connected component labeling algorithms",
  Paper LBNL-56864, 2005,
  Lawrence Berkeley National Laboratory
  (University of California),
  http://repositories.cdlib.org/lbnl/LBNL-56864

"""

DTYPE = np.intp

# Short int - could be more graceful to the CPU cache
ctypedef cnp.int32_t INTS_t

cdef struct s_shpinfo

ctypedef s_shpinfo shape_info
ctypedef int (* fun_ravel)(int, int, int, shape_info *)


# For having stuff concerning background in one place
ctypedef struct bginfo:
    ## The value in the image (i.e. not the label!) that identifies
    ## the background.
    DTYPE_t background_val
    DTYPE_t background_node
    ## Identification of the background in the labelled image
    DTYPE_t background_label


cdef void get_bginfo(background_val, bginfo *ret) except *:
    if background_val is None:
        warnings.warn(DeprecationWarning(
                'The default value for `background` will change to 0 in v0.12'
            ))
        ret.background_val = -1
    else:
        ret.background_val = background_val

    # The node -999 doesn't exist, it will get subsituted by a meaningful value
    # upon the first background pixel occurence
    ret.background_node = -999
    ret.background_label = -1


# A pixel has neighbors that have already been scanned.
# In the paper, the pixel is denoted by E and its neighbors:
#
#    z=1        z=0       x
#    ---------------------->
#   | A B C      F G H
#   | D E .      I J K
#   | . . .      L M N
#   |
# y V
#
# D_ea represents offset of A from E etc. - see the definition of
# get_shape_info
cdef enum:
    # the 0D neighbor
    # D_ee, # We don't need D_ee
    # the 1D neighbor
    D_ed,
    # 2D neighbors
    D_ea, D_eb, D_ec,
    # 3D neighbors
    D_ef, D_eg, D_eh, D_ei, D_ej, D_ek, D_el, D_em, D_en,
    D_COUNT


# Structure for centralised access to shape data
# Contains information related to the shape of the input array
cdef struct s_shpinfo:
    INTS_t x
    INTS_t y
    INTS_t z

    # Number of elements
    DTYPE_t numels
    # Number of of the input array
    INTS_t ndim

    # Offsets between elements recalculated to linear index increments
    # DEX[D_ea] is offset between E and A (i.e. to the point to upper left)
    # The name DEX is supposed to evoke DE., where . = A, B, C, D, F etc.
    INTS_t DEX[D_COUNT]

    # Function pointer to a function that recalculates multiindex to linear
    # index. Heavily depends on dimensions of the input array.
    fun_ravel ravel_index


cdef void get_shape_info(inarr_shape, shape_info *res) except *:
    """
    Precalculates all the needed data from the input array shape
    and stores them in the shape_info struct.
    """
    res.y = 1
    res.z = 1
    res.ravel_index = ravel_index2D

    res.ndim = len(inarr_shape)
    if res.ndim == 1:
        res.x = inarr_shape[0]
        res.ravel_index = ravel_index1D
    elif res.ndim == 2:
        res.x = inarr_shape[1]
        res.y = inarr_shape[0]
        res.ravel_index = ravel_index2D
    elif res.ndim == 3:
        res.x = inarr_shape[2]
        res.y = inarr_shape[1]
        res.z = inarr_shape[0]
        res.ravel_index = ravel_index3D
    else:
        raise NotImplementedError(
            "Only for images of dimension 1-3 are supported, got a %sD one"
            % res.ndim)

    res.numels = res.x * res.y * res.z

    # When reading this for the first time, look at the diagram by the enum
    # definition above (keyword D_ee)
    # Difference between E and G is (x=0, y=-1, z=-1), E and A (-1, -1, 0) etc.
    # Here, it is recalculated to linear (raveled) indices of flattened arrays
    # with their last (=contiguous) dimension is x.

    # So now the 1st (needed for 1D, 2D and 3D) part, y = 1, z = 1
    res.DEX[D_ed] = -1

    # Not needed, just for illustration
    # res.DEX[D_ee] = 0

    # So now the 2nd (needed for 2D and 3D) part, y = 0, z = 1
    res.DEX[D_ea] = res.ravel_index(-1, -1, 0, res)
    res.DEX[D_eb] = res.DEX[D_ea] + 1
    res.DEX[D_ec] = res.DEX[D_eb] + 1

    # And now the 3rd (needed only for 3D) part, z = 0
    res.DEX[D_ef] = res.ravel_index(-1, -1, -1, res)
    res.DEX[D_eg] = res.DEX[D_ef] + 1
    res.DEX[D_eh] = res.DEX[D_ef] + 2
    res.DEX[D_ei] = res.DEX[D_ef] - res.DEX[D_eb]  # DEX[D_eb] = one row up, remember?
    res.DEX[D_ej] = res.DEX[D_ei] + 1
    res.DEX[D_ek] = res.DEX[D_ei] + 2
    res.DEX[D_el] = res.DEX[D_ei] - res.DEX[D_eb]
    res.DEX[D_em] = res.DEX[D_el] + 1
    res.DEX[D_en] = res.DEX[D_el] + 2


cdef inline void join_trees_wrapper(DTYPE_t *data_p, DTYPE_t *forest_p,
                                    DTYPE_t rindex, INTS_t idxdiff):
    if data_p[rindex] == data_p[rindex + idxdiff]:
        join_trees(forest_p, rindex, rindex + idxdiff)


cdef int ravel_index1D(int x, int y, int z, shape_info *shapeinfo):
    """
    Ravel index of a 1D array - trivial. y and z are ignored.
    """
    return x


cdef int ravel_index2D(int x, int y, int z, shape_info *shapeinfo):
    """
    Ravel index of a 2D array. z is ignored
    """
    cdef int ret = x + y * shapeinfo.x
    return ret


cdef int ravel_index3D(int x, int y, int z, shape_info *shapeinfo):
    """
    Ravel index of a 3D array
    """
    cdef int ret = x + y * shapeinfo.x + z * shapeinfo.y * shapeinfo.x
    return ret


# Tree operations implemented by an array as described in Wu et al.
# The term "forest" is used to indicate an array that stores one or more trees

# Consider a following tree:
#
# 5 ----> 3 ----> 2 ----> 1 <---- 6 <---- 7
#                 |               |
#          4 >----/               \----< 8 <---- 9
#
# The vertices are a unique number, so the tree can be represented by an
# array where a the tuple (index, array[index]) represents an edge,
# so for our example, array[2] == 1, array[7] == 6 and array[1] == 1, because
# 1 is the root.
# Last but not least, one array can hold more than one tree as long as their
# indices are different. It is the case in this algorithm, so for that reason
# the array is referred to as the "forest" = multiple trees next to each
# other.
#
# In this algorithm, there are as many indices as there are elements in the
# array to label and array[x] == x for all x. As the labelling progresses,
# equivalence between so-called provisional (i.e. not final) labels is
# discovered and trees begin to surface.
# When we found out that label 5 and 3 are the same, we assign array[5] = 3.

cdef DTYPE_t find_root(DTYPE_t *forest, DTYPE_t n):
    """Find the root of node n.
    Given the example above, for any integer from 1 to 9, 1 is always returned
    """
    cdef DTYPE_t root = n
    while (forest[root] < root):
        root = forest[root]
    return root


cdef inline void set_root(DTYPE_t *forest, DTYPE_t n, DTYPE_t root):
    """
    Set all nodes on a path to point to new_root.
    Given the example above, given n=9, root=6, it would "reconnect" the tree.
    so forest[9] = 6 and forest[8] = 6
    The ultimate goal is that all tree nodes point to the real root,
    which is element 1 in this case.
    """
    cdef DTYPE_t j
    while (forest[n] < n):
        j = forest[n]
        forest[n] = root
        n = j

    forest[n] = root


cdef inline void join_trees(DTYPE_t *forest, DTYPE_t n, DTYPE_t m):
    """Join two trees containing nodes n and m.
    If we imagine that in the example tree, the root 1 is not known, we
    rather have two disjoint trees with roots 2 and 6.
    Joining them would mean that all elements of both trees become connected
    to the element 2, so forest[9] == 2, forest[6] == 2 etc.
    However, when the relationship between 1 and 2 can still be discovered later.
    """
    cdef DTYPE_t root
    cdef DTYPE_t root_m

    if (n != m):
        root = find_root(forest, n)
        root_m = find_root(forest, m)

        if (root > root_m):
            root = root_m

        set_root(forest, n, root)
        set_root(forest, m, root)


def _norm_connectivity(connectivity, ndim):
    """
    Takes the value of the connectivity parameter, validates it and converts
    it to a value that the subsequent algorithm may use as-is safely.

    Parameters
    ----------
    connectivity : int
        The following should be true: -ndim < connectivity < 0, or
        0 < connectivity <= ndim.

    Returns
    -------
    connectivity : int
        Connectivity, 0 < connectivity < ndim
    """
    if connectivity == 0 or connectivity > ndim:
        raise ValueError(
            "Connectivity of 0 or above %d doesn't make sense" % ndim)
    res = connectivity
    if res < 0:
        res = res % ndim
        # we want -1 to be normed to ndim, -2 to (ndim - 1) etc.
        res += 1
    return res


# Connected components search as described in Fiorio et al.
def label(input, neighbors=None, background=None, return_num=False,
          connectivity=None):
    """Label connected regions of an integer array.

    Two pixels are connected when they are neighbors and have the same value.
    In 2D, they can be neighbors either in a 1- or 2-connected sense.
    The value refers to the maximum number of orthogonal hops to consider a
    pixel/voxel a neighbor.

      1-connectivity      2-connectivity     diagonal connection close-up

           [ ]           [ ]  [ ]  [ ]         [ ]
            |               \\ |  /             |  <- hop 2
      [ ]--[x]--[ ]      [ ]--[x]--[ ]    [x]--[ ]
            |               /  |  \\        hop 1
           [ ]           [ ]  [ ]  [ ]

    Parameters
    ----------
    input : ndarray of dtype int
        Image to label.
    neighbors : {4, 8}, int, optional
        Whether to use 4- or 8-connectivity.
        In 3D, 4-connectivity means connected pixels have to share face,
        whereas with 8-connectivity, they have to share only edge or vertex.
        **Deprecated, use ``connectivity`` instead.**
    background : int, optional
        Consider all pixels with this value as background pixels, and label
        them as -1. (Note: background pixels will be labeled as 0 starting with
        version 0.12).
    return_num : bool, optional
        Whether to return the number of assigned labels.
    connectivity : int, optional
        Number of orthogonal hops
        For the 2D case, 1 considers horizontal and vertical neighbors, whereas
        2 adds the diagonals (you hop once vertically and once horizontally).
        1 is the lowest value of connection (4 neighbors in 2D, 6 in 3D).
        Moreover, the value of -1 specifies the highest connectivity available.
        So for example in 2D, -1 is equivalent of 2, resulting in considering
        all 8 neighbors.

    Returns
    -------
    labels : ndarray of dtype int
        Labeled array, where all connected regions are assigned the
        same integer value.
    num : int, optional
        Number of labels, which equals the maximum label index and is only
        returned if return_num is `True`.

    Examples
    --------
    >>> x = np.eye(3).astype(int)
    >>> print(x)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]

    >>> print(m.label(x, neighbors=4))
    [[0 1 1]
     [2 3 1]
     [2 2 4]]

    >>> print(m.label(x, neighbors=8))
    [[0 1 1]
     [1 0 1]
     [1 1 0]]

    >>> x = np.array([[1, 0, 0],
    ...               [1, 1, 5],
    ...               [0, 0, 0]])

    >>> print(m.label(x, background=0))
    [[ 0 -1 -1]
     [ 0  0  1]
     [-1 -1 -1]]

    """

    cdef cnp.ndarray[DTYPE_t, ndim=1] data
    cdef cnp.ndarray[DTYPE_t, ndim=1] forest

    # Having data a 2D array slows down access considerably using linear
    # indices even when using the data_p pointer :-(
    data = np.copy(input.flatten().astype(DTYPE))
    forest = np.arange(data.size, dtype=DTYPE)

    cdef DTYPE_t *forest_p = <DTYPE_t*>forest.data
    cdef DTYPE_t *data_p = <DTYPE_t*>data.data

    cdef shape_info shapeinfo
    cdef bginfo bg

    get_shape_info(input.shape, &shapeinfo)
    get_bginfo(background, &bg)

    if neighbors is None and connectivity is None:
        # default
        connectivity = -1
    elif neighbors is not None:
        depr_msg = ("The argument 'neighbors' is deprecated, use 'connectivity'"
                    " instead")
        # fail
        if neighbors != 4 and neighbors != 8:
            DeprecationWarning(depr_msg)
            msg = "Neighbors must be either 4 or 8, got '%d'.\n" % neighbors
            raise ValueError(msg)
        else:
            # backwards-compatible neighbors recalc to connectivity,
            # deprecation warning
            nei2conn = {4: 1, 8: -1}
            connectivity = nei2conn[neighbors]
            msg = " Its corresponing value is '%d'" % connectivity
            DeprecationWarning(depr_msg + msg)

    connectivity = _norm_connectivity(connectivity, shapeinfo.ndim)

    scanBG(data_p, forest_p, &shapeinfo, &bg)
    scan3D(data_p, forest_p, &shapeinfo, &bg, connectivity)

    # Label output
    cdef DTYPE_t ctr
    ctr = resolve_labels(data_p, forest_p, &shapeinfo, &bg)

    # Work around a bug in ndimage's type checking on 32-bit platforms
    if data.dtype == np.int32:
        data = data.view(np.int32)

    res = data.reshape(input.shape)

    if return_num:
        return res, ctr
    else:
        return res


cdef DTYPE_t resolve_labels(DTYPE_t *data_p, DTYPE_t *forest_p,
                            shape_info *shapeinfo, bginfo *bg):
    """
    We iterate through the provisional labels and assign final labels based on
    our knowledge of prov. labels relationship.
    We also track how many distinct final labels we have.
    """
    cdef DTYPE_t counter = bg.background_label + 1, i

    for i in range(shapeinfo.numels):
        if i == bg.background_node:
            data_p[i] = bg.background_label
        elif i == forest_p[i]:
            # We have stumbled across a root which is something new to us (root
            # is the LOWEST of all prov. labels that are equivalent to it)
            data_p[i] = counter
            counter += 1
        else:
            data_p[i] = data_p[forest_p[i]]
    return counter


cdef void scanBG(DTYPE_t *data_p, DTYPE_t *forest_p, shape_info *shapeinfo,
                 bginfo *bg):
    """
    Settle all background pixels now and don't bother with them later.
    Since this only requires one linar sweep through the array, it is fast
    and it makes sense to do it separately.

    The result of this function is update of forest_p and bg parameter.
    """
    cdef DTYPE_t i, bgval = bg.background_val, firstbg
    # We find the provisional label of the background, which is the index of
    # the first background pixel
    for i in range(shapeinfo.numels):
        if data_p[i] == bgval:
            firstbg = i
            bg.background_node = firstbg
            break

    # And then we apply this provisional label to the whole background
    for i in range(firstbg, shapeinfo.numels):
        if data_p[i] == bgval:
            forest_p[i] = firstbg


# Here, we work with flat arrays regardless whether the data is 1, 2 or 3D.
# The lookup to the neighbor in a 2D array is achieved by precalculating an
# offset and adding it to the index.
# The forward scan mask looks like this (the center point is actually E):
# (take a look at shape_info docs for more exhaustive info)
# A B C
# D E
#
# So if I am in the point E and want to take a look to A, I take the index of
# E and add shapeinfo.DEX[D_ea] to it and get the index of A.
# The 1D indices are "raveled" or "linear", that's where "rindex" comes from.


cdef void scan1D(DTYPE_t *data_p, DTYPE_t *forest_p, shape_info *shapeinfo,
                 bginfo *bg, DTYPE_t connectivity, DTYPE_t y, DTYPE_t z):
    """
    Perform forward scan on a 1D object, usually the first row of an image
    """
    # Initialize the first row
    cdef DTYPE_t x, rindex, bgval = bg.background_val
    cdef INTS_t *DEX = shapeinfo.DEX
    rindex = shapeinfo.ravel_index(0, y, z, shapeinfo)

    for x in range(1, shapeinfo.x):
        rindex += 1
        # Handle the first row
        if data_p[rindex] == bgval:
            # Nothing to do if we are background
            continue

        join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ed])


cdef void scan2D(DTYPE_t *data_p, DTYPE_t *forest_p, shape_info *shapeinfo,
                 bginfo *bg, DTYPE_t connectivity, DTYPE_t z):
    """
    Perform forward scan on a 2D array.
    """
    cdef DTYPE_t x, y, rindex, bgval = bg.background_val
    cdef INTS_t *DEX = shapeinfo.DEX
    scan1D(data_p, forest_p, shapeinfo, bg, connectivity, 0, z)
    for y in range(1, shapeinfo.y):
        # BEGINNING of x = 0
        rindex = shapeinfo.ravel_index(0, y, 0, shapeinfo)
        # Handle the first column
        if data_p[rindex] != bgval:
            # Nothing to do if we are background

            join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eb])

            if connectivity >= 2:
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ec])
        # END of x = 0

        for x in range(1, shapeinfo.x - 1):
            # We have just moved to another column (of the same row)
            # so we increment the raveled index. It will be reset when we get
            # to another row, so we don't have to worry about altering it here.
            rindex += 1
            if data_p[rindex] == bgval:
                # Nothing to do if we are background
                continue

            join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eb])
            join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ed])

            if connectivity >= 2:
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ea])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ec])

        # Finally, the last column
        # BEGINNING of x = max
        rindex += 1
        if data_p[rindex] != bgval:
            # Nothing to do if we are background

            join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eb])
            join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ed])

            if connectivity >= 2:
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ea])
        # END of x = max


cdef void scan3D(DTYPE_t *data_p, DTYPE_t *forest_p, shape_info *shapeinfo,
                 bginfo *bg, DTYPE_t connectivity):
    """
    Perform forward scan on a 2D array.

    """
    cdef DTYPE_t x, y, z, rindex, bgval = bg.background_val
    cdef INTS_t *DEX = shapeinfo.DEX
    # Handle first plane
    scan2D(data_p, forest_p, shapeinfo, bg, connectivity, 0)
    for z in range(1, shapeinfo.z):
        # Handle first row in 3D manner
        # BEGINNING of y = 0
        # BEGINNING of x = 0
        rindex = shapeinfo.ravel_index(0, 0, z, shapeinfo)
        if data_p[rindex] != bgval:
            # Nothing to do if we are background

            # Now we have pixels below
            join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ej])

            if connectivity >= 2:
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ek])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_em])
                if connectivity >= 3:
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_en])
        # END of x = 0

        for x in range(1, shapeinfo.x - 1):
            rindex += 1
            # Handle the first row
            if data_p[rindex] == bgval:
                # Nothing to do if we are background
                continue

            join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ed])
            join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ej])

            if connectivity >= 2:
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ei])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ek])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_em])
                if connectivity >= 3:
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_el])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_en])

        # BEGINNING of x = max
        rindex += 1
        # Handle the last element of the first row
        if data_p[rindex] != bgval:
            # Nothing to do if we are background

            join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ed])
            join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ej])

            if connectivity >= 2:
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ei])
                if connectivity >= 3:
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_el])
        # END of x = max
        # END of y = 0

        for y in range(1, shapeinfo.y - 1):
            # BEGINNING of y = ...
            # BEGINNING of x = 0
            rindex = shapeinfo.ravel_index(0, y, z, shapeinfo)
            # Handle the first column in 3D manner
            if data_p[rindex] != bgval:
                # Nothing to do if we are background

                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eb])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ej])

                if connectivity >= 2:
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ec])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eg])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ek])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_em])
                    if connectivity >= 3:
                        join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eh])
                        join_trees_wrapper(data_p, forest_p, rindex, DEX[D_en])
            # END of x = 0

            # Handle the rest of columns
            for x in range(1, shapeinfo.x - 1):
                rindex += 1
                if data_p[rindex] == bgval:
                    # Nothing to do if we are background
                    continue

                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eb])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ed])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ej])

                if connectivity >= 2:
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ea])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eg])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ei])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ec])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ek])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_em])
                    if connectivity >= 3:
                        join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ef])
                        join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eh])
                        join_trees_wrapper(data_p, forest_p, rindex, DEX[D_el])
                        join_trees_wrapper(data_p, forest_p, rindex, DEX[D_en])

            # BEGINNING of x = max
            rindex += 1
            if data_p[rindex] != bgval:
                # Nothing to do if we are background

                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eb])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ed])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ej])

                if connectivity >= 2:
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ea])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eg])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ei])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_em])
                    if connectivity >= 3:
                        join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ef])
                        join_trees_wrapper(data_p, forest_p, rindex, DEX[D_el])
            # END of x = max
            # END of y = ...

            # BEGINNING of y = max
            # BEGINNING of x = 0
            rindex = shapeinfo.ravel_index(0, shapeinfo.y, z, shapeinfo)
            # Handle the first column in 3D manner
            if data_p[rindex] != bgval:
                # Nothing to do if we are background

                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eb])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ej])

                if connectivity >= 2:
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ec])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eg])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ek])
                    if connectivity >= 3:
                        join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eh])
            # END of x = 0

            # Handle the rest of columns
            for x in range(1, shapeinfo.x - 1):
                rindex += 1
                if data_p[rindex] == bgval:
                    # Nothing to do if we are background
                    continue

                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eb])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ed])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ej])

                if connectivity >= 2:
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ea])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eg])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ei])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ec])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ek])
                    if connectivity >= 3:
                        join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ef])
                        join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eh])

            # BEGINNING of x = max
            rindex += 1
            if data_p[rindex] != bgval:
                # Nothing to do if we are background

                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eb])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ed])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ej])

                if connectivity >= 2:
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ea])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eg])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ei])
                    if connectivity >= 3:
                        join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ef])
            # END of x = max
            # END of y = max
