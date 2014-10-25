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

ctypedef s_shpinfo shpinfo
ctypedef s_bginfo bginfo
ctypedef int (* fun_ravel)(int, int, int, shpinfo *)


cdef struct s_bginfo:
    DTYPE_t background_val
    DTYPE_t background_node


# Structure for centralised access to shape data
cdef struct s_shpinfo:
    INTS_t x
    INTS_t y
    INTS_t z

    DTYPE_t numels
    INTS_t ndim

    #INTS_t Dee
    INTS_t Ded

    INTS_t Dea
    INTS_t Deb
    INTS_t Dec

    INTS_t Def
    INTS_t Deg
    INTS_t Deh
    INTS_t Dei
    INTS_t Dej
    INTS_t Dek
    INTS_t Del
    INTS_t Dem
    INTS_t Den

    fun_ravel ravel_index


cdef shpinfo get_triple(inarr_shape):
    cdef shpinfo res = shpinfo()

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
        assert "Only for images of dimension 1-3 (got %s)" % res.ndim

    res.numels = res.x * res.y * res.z

    # Our point of interest is E.
    #    z=1        z=0       x
    #    ---------------------->
    #   | A B C      F G H
    #   | D E .      I J K
    #   | . . .      L M N
    #   |
    # y V
    #
    # Difference between E and G is (x=0, y=-1, z=-1), E and A (-1, -1, 0) etc.
    # Here, it is recalculated to linear (raveled) indices of flattened arrays
    # with their last (=contiguous) dimension is x.

    # So now the 1st (needed for 1D, 2D and 3D) part, y = 1, z = 1
    res.Ded = - 1

    # Not needed, just for illustration
    # + enabling it prolongs the exec time quite considerably - why?
    #res.Dee = 0

    # So now the 2nd (needed for 2D and 3D) part, y = 0, z = 1
    res.Dea = res.ravel_index(-1, -1, 0, & res)
    res.Deb = res.Dea + 1
    res.Dec = res.Deb + 1

    # And now the 3rd (needed only for 3D) part, z = 0
    res.Def = res.ravel_index(-1, -1, -1, & res)
    res.Deg = res.Def + 1
    res.Deh = res.Def + 2
    res.Dei = res.Def - res.Deb  # Deb = one row up, remember?
    res.Dej = res.Dei + 1
    res.Dek = res.Dei + 2
    res.Del = res.Dei - 2 * res.Deb
    res.Dem = res.Del + 1
    res.Den = res.Del + 2

    return res


cdef int ravel_index1D(int x, int y, int z, shpinfo * shapeinfo):
    """
    Ravel index of a 1D array - trivial. y and z are ignored.
    """
    return x


cdef int ravel_index2D(int x, int y, int z, shpinfo * shapeinfo):
    """
    Ravel index of a 2D array. z is ignored
    """
    cdef int ret = x + y * shapeinfo.x
    return ret


cdef int ravel_index3D(int x, int y, int z, shpinfo * shapeinfo):
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
# the array is referred to as the "forrest" = multiple trees next to each
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


cdef inline void link_bg(DTYPE_t *forest, DTYPE_t n, DTYPE_t *background_node):
    """
    Link a node to the background node.

    """
    if background_node[0] == -999:
        background_node[0] = n

    join_trees(forest, n, background_node[0])


# Connected components search as described in Fiorio et al.
def label(input, DTYPE_t neighbors=8, background=None, return_num=False):
    """Label connected regions of an integer array.

    Two pixels are connected when they are neighbors and have the same value.
    They can be neighbors either in a 4- or 8-connected sense::

      4-connectivity      8-connectivity

           [ ]           [ ]  [ ]  [ ]
            |               \  |  /
      [ ]--[ ]--[ ]      [ ]--[ ]--[ ]
            |               /  |  \\
           [ ]           [ ]  [ ]  [ ]

    Parameters
    ----------
    input : ndarray of dtype int
        Image to label.
    neighbors : {4, 8}, int, optional
        Whether to use 4- or 8-connectivity.
    background : int, optional
        Consider all pixels with this value as background pixels, and label
        them as -1. (Note: background pixels will be labeled as 0 starting with
        version 0.12).
    return_num : bool, optional
        Whether to return the number of assigned labels.

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
    data = input.flatten().astype(DTYPE, copy=True)
    forest = np.arange(data.size, dtype=DTYPE)

    cdef DTYPE_t *forest_p = <DTYPE_t*>forest.data
    cdef DTYPE_t *data_p = <DTYPE_t*>data.data

    cdef shpinfo shapeinfo
    cdef bginfo bg

    shapeinfo = get_triple(input.shape)

    bg.background_val = 0
    bg.background_node = -999

    if background is None:
        bg.background_val = -1
        warnings.warn(DeprecationWarning(
                'The default value for `background` will change to 0 in v0.12'
            ))
    else:
        bg.background_val = background

    if neighbors != 4 and neighbors != 8:
        raise ValueError('Neighbors must be either 4 or 8.')

    if shapeinfo.ndim == 1:
        scan1D(data_p, forest_p, & shapeinfo, & bg, neighbors, 0, 0)
    elif shapeinfo.ndim == 2:
        scan2D(data_p, forest_p, & shapeinfo, & bg, neighbors, 0)

    # Label output
    cdef DTYPE_t ctr = 0
    ctr = resolve_labels(data_p, forest_p, & shapeinfo, & bg)

    # Work around a bug in ndimage's type checking on 32-bit platforms
    if data.dtype == np.int32:
        data = data.view(np.int32)

    res = data.reshape(input.shape)

    if return_num:
        return res, ctr
    else:
        return res


cdef DTYPE_t resolve_labels(DTYPE_t * data_p, DTYPE_t * forest_p,
                            shpinfo * shapeinfo, bginfo * bg):
    """
    We iterate through the provisional labels and assign final labels based on
    our knowledge of prov. labels relationship.
    We also track how many distinct final labels we have.
    """
    cdef DTYPE_t counter = 0
    for i in range(shapeinfo.numels):
        if i == bg.background_node:
            data_p[i] = -1
        elif i == forest_p[i]:
            data_p[i] = counter
            counter += 1
        else:
            data_p[i] = data_p[forest_p[i]]
    return counter


# Here, we work with flat arrays regardless whether the data is 1, 2 or 3D.
# The lookup to the neighbor in a 2D array is achieved by precalculating an
# offset and ading it to the index.
# The forward scan mask looks like this (the center point is actually E):
# (take a look at shpinfo docs for more exhaustive info)
# A B C
# D E
#
# So if I am in the point E and want to take a look to A, I take the index of
# E and add shapeinfo.Dea to it and teg the index of A.
# The 1D indices are "raveled" or "linear", that's where "rindex" comes from.


cdef scan1D(DTYPE_t * data_p, DTYPE_t * forest_p, shpinfo * shapeinfo,
            bginfo * bg, DTYPE_t neighbors, DTYPE_t y, DTYPE_t z):
    """
    Perform forward scan on a 1D object, usually the first row of an image
    """
    # Initialize the first row
    cdef DTYPE_t x, rindex
    rindex = shapeinfo.ravel_index(0, y, z, shapeinfo)

    if data_p[rindex] == bg.background_val:
        link_bg(forest_p, rindex, & bg.background_node)

    for x in range(1, shapeinfo.x):
        rindex += 1
        # Handle the first row
        # First row => rindex == j
        if data_p[rindex] == bg.background_val:
            link_bg(forest_p, rindex, & bg.background_node)

        if data_p[rindex] == data_p[rindex + shapeinfo.Ded]:
            join_trees(forest_p, rindex, rindex + shapeinfo.Ded)


cdef scan2D(DTYPE_t * data_p, DTYPE_t * forest_p, shpinfo * shapeinfo,
            bginfo * bg, DTYPE_t neighbors, DTYPE_t z):
    """
    Perform forward scan on a 2D array.
    """
    cdef DTYPE_t x, y, rindex
    scan1D(data_p, forest_p, shapeinfo, bg, neighbors, 0, z)
    for y in range(1, shapeinfo.y):
        rindex = shapeinfo.ravel_index(0, y, 0, shapeinfo)
        # Handle the first column
        if data_p[rindex] == bg.background_val:
            link_bg(forest_p, rindex, & bg.background_node)

        if data_p[rindex] == data_p[rindex + shapeinfo.Deb]:
            join_trees(forest_p, rindex, rindex + shapeinfo.Deb)

        if neighbors == 8:
            if data_p[rindex] == data_p[rindex + shapeinfo.Dec]:
                join_trees(forest_p, rindex, rindex + shapeinfo.Dec)

        # Handle the rest of columns
        for x in range(1, shapeinfo.x):
            # We have just moved to another column (of the same row)
            # so we increment the raveled index. It will be reset when we get
            # to another row, so we don't have to worry about altering it here.
            rindex += 1
            if data_p[rindex] == bg.background_val:
                link_bg(forest_p, rindex, & bg.background_node)

            if neighbors == 8:
                if data_p[rindex] == data_p[rindex + shapeinfo.Dea]:
                    join_trees(forest_p, rindex, rindex + shapeinfo.Dea)

            if data_p[rindex] == data_p[rindex + shapeinfo.Deb]:
                join_trees(forest_p, rindex, rindex + shapeinfo.Deb)

            if neighbors == 8:
                if x < shapeinfo.x - 1:
                    if data_p[rindex] == data_p[rindex + shapeinfo.Dec]:
                        join_trees(forest_p, rindex, rindex + shapeinfo.Dec)

            if data_p[rindex] == data_p[rindex + shapeinfo.Ded]:
                join_trees(forest_p, rindex, rindex + shapeinfo.Ded)
