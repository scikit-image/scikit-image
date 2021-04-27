#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
from warnings import warn

cimport numpy as cnp
cnp.import_array()

DTYPE = np.intp
cdef DTYPE_t BG_NODE_NULL = -999

cdef struct s_shpinfo

ctypedef s_shpinfo shape_info
ctypedef size_t (* fun_ravel)(size_t, size_t, size_t, shape_info *) nogil


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
        ret.background_val = 0
    else:
        ret.background_val = background_val

    # The node -999 doesn't exist, it will get subsituted by a meaningful value
    # upon the first background pixel occurrence
    ret.background_node = BG_NODE_NULL
    ret.background_label = 0


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


# Structure for centralized access to shape data
# Contains information related to the shape of the input array
cdef struct s_shpinfo:
    DTYPE_t x
    DTYPE_t y
    DTYPE_t z

    # Number of elements
    DTYPE_t numels
    # Dimensions of of the input array
    DTYPE_t ndim

    # Offsets between elements recalculated to linear index increments
    # DEX[D_ea] is offset between E and A (i.e. to the point to upper left)
    # The name DEX is supposed to evoke DE., where . = A, B, C, D, F etc.
    DTYPE_t DEX[D_COUNT]

    # Function pointer to a function that recalculates multi-index to linear
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
    # A shape (3, 1, 4) would make the algorithm crash, but the corresponding
    # good_shape (i.e. the array with axis swapped) (1, 3, 4) is OK.
    # Having an axis length of 1 when an axis on the left is longer than 1
    # (in this case, it has length of 3) is NOT ALLOWED.
    good_shape = tuple(sorted(inarr_shape))

    res.ndim = len(inarr_shape)
    if res.ndim == 1:
        res.x = inarr_shape[0]
        res.ravel_index = ravel_index1D
    elif res.ndim == 2:
        res.x = inarr_shape[1]
        res.y = inarr_shape[0]
        res.ravel_index = ravel_index2D
        if res.x == 1 and res.y > 1:
            # Should not occur, but better be safe than sorry
            raise ValueError(
                "Swap axis of your %s array so it has a %s shape"
                % (inarr_shape, good_shape))
    elif res.ndim == 3:
        res.x = inarr_shape[2]
        res.y = inarr_shape[1]
        res.z = inarr_shape[0]
        res.ravel_index = ravel_index3D
        if ((res.x == 1 and res.y > 1)
            or res.y == 1 and res.z > 1):
            # Should not occur, but better be safe than sorry
            raise ValueError(
                "Swap axes of your %s array so it has a %s shape"
                % (inarr_shape, good_shape))
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
                                    DTYPE_t rindex, DTYPE_t idxdiff) nogil:
    if data_p[rindex] == data_p[rindex + idxdiff]:
        join_trees(forest_p, rindex, rindex + idxdiff)


cdef size_t ravel_index1D(size_t x, size_t y, size_t z,
                          shape_info *shapeinfo) nogil:
    """
    Ravel index of a 1D array - trivial. y and z are ignored.
    """
    return x


cdef size_t ravel_index2D(size_t x, size_t y, size_t z,
                          shape_info *shapeinfo) nogil:
    """
    Ravel index of a 2D array. z is ignored
    """
    cdef size_t ret = x + y * shapeinfo.x
    return ret


cdef size_t ravel_index3D(size_t x, size_t y, size_t z,
                          shape_info *shapeinfo) nogil:
    """
    Ravel index of a 3D array
    """
    cdef size_t ret = x + y * shapeinfo.x + z * shapeinfo.y * shapeinfo.x
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

cdef DTYPE_t find_root(DTYPE_t *forest, DTYPE_t n) nogil:
    """Find the root of node n.
    Given the example above, for any integer from 1 to 9, 1 is always returned
    """
    cdef DTYPE_t root = n
    while (forest[root] < root):
        root = forest[root]
    return root


cdef inline void set_root(DTYPE_t *forest, DTYPE_t n, DTYPE_t root) nogil:
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


cdef inline void join_trees(DTYPE_t *forest, DTYPE_t n, DTYPE_t m) nogil:
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


def _get_swaps(shp):
    """
    What axes to swap if we want to convert an illegal array shape
    to a legal one.

    Args:
        shp (tuple-like): The array shape

    Returns:
        list: List of tuples to be passed to np.swapaxes
    """
    shp = np.array(shp)
    swaps = []

    # Dimensions where the array is "flat"
    ones = np.where(shp == 1)[0][::-1]
    # The other dimensions
    bigs = np.where(shp > 1)[0]
    # We now go from opposite directions and swap axes if an index of a flat
    # axis is higher than the one of a thick axis
    for one, big in zip(ones, bigs):
        if one < big:
            # We are ordered already
            break
        else:
            swaps.append((one, big))
    # TODO: Add more swaps so the array is longer along x and shorter along y
    return swaps


def _apply_swaps(arr, swaps):
    arr2 = arr
    for one, two in swaps:
        arr2 = arr.swapaxes(one, two)
    return arr2


def reshape_array(arr):
    """
    "Rotates" the array so it gains a shape that the algorithm can work with.
    An illegal shape is (3, 1, 4), and correct one is (1, 3, 4) or (1, 4, 3).
    The point is to have all 1s of the shape at the beginning, not in the
    middle of the shape tuple.

    Note: The greater-than-one shape component should go from greatest to
    lowest numbers since it is more friendly to the CPU cache (so (1, 3, 4) is
    less optimal than (1, 4, 3). Keyword to this is "memory spatial locality"

    Args:
        arr (np.ndarray): The array we want to fix

    Returns:
        tuple (result, swaps): The result is the "fixed" array and a record
        of what has been done with it.
    """
    swaps = _get_swaps(arr.shape)
    reshaped = _apply_swaps(arr, swaps)
    return reshaped, swaps


def undo_reshape_array(arr, swaps):
    """
    Does the opposite of what :func:`reshape_array` does

    Args:
        arr (np.ndarray): The array to "revert"
        swaps (list): The second result of :func:`reshape_array`

    Returns:
        np.ndarray: The array of the original shape
    """
    # It is safer to undo axes swaps in the opposite order
    # than the application order
    reshaped = _apply_swaps(arr, swaps[::-1])
    return reshaped


def label_cython(input_, background=None, return_num=False,
                 connectivity=None):
    # Connected components search as described in Fiorio et al.
    # We have to ensure that the shape of the input can be handled by the
    # algorithm.  The input is reshaped as needed for compatibility.
    input_, swaps = reshape_array(input_)
    shape = input_.shape
    ndim = input_.ndim

    cdef cnp.ndarray[DTYPE_t, ndim=1] forest

    # Having data a 2D array slows down access considerably using linear
    # indices even when using the data_p pointer :-(

    # np.array makes a copy so it is safe to modify data in-place
    data = np.array(input_, order='C', dtype=DTYPE)
    forest = np.arange(data.size, dtype=DTYPE)

    cdef DTYPE_t *forest_p = <DTYPE_t*>forest.data
    cdef DTYPE_t *data_p = <DTYPE_t*>cnp.PyArray_DATA(data)

    cdef shape_info shapeinfo
    cdef bginfo bg

    get_shape_info(shape, &shapeinfo)
    get_bginfo(background, &bg)

    if connectivity is None:
        # use the full connectivity by default
        connectivity = ndim

    if not 1 <= connectivity <= ndim:
        raise ValueError(
            f'Connectivity for {input_.ndim}D image should '
            f'be in [1, ..., {input_.ndim}]. Got {connectivity}.'
        )

    cdef DTYPE_t conn = connectivity
    # Label output
    cdef DTYPE_t ctr
    with nogil:
        scanBG(data_p, forest_p, &shapeinfo, &bg)
        # the data are treated as degenerated 3D arrays if needed
        # without any performance sacrifice
        scan3D(data_p, forest_p, &shapeinfo, &bg, conn)
        ctr = resolve_labels(data_p, forest_p, &shapeinfo, &bg)

    # Work around a bug in ndimage's type checking on 32-bit platforms
    if data.dtype == np.int32:
        data = data.view(np.int32)

    if swaps:
        data = undo_reshape_array(data, swaps)

    if return_num:
        return data, ctr
    else:
        return data


cdef DTYPE_t resolve_labels(DTYPE_t *data_p, DTYPE_t *forest_p,
                            shape_info *shapeinfo, bginfo *bg) nogil:
    """
    We iterate through the provisional labels and assign final labels based on
    our knowledge of prov. labels relationship.
    We also track how many distinct final labels we have.
    """
    cdef DTYPE_t counter = 1, i

    for i in range(shapeinfo.numels):
        if i == forest_p[i]:
            # We have stumbled across a root which is something new to us (root
            # is the LOWEST of all prov. labels that are equivalent to it)

            # If the root happens to be the background,
            # assign the background label instead of a
            # new label from the counter
            if i == bg.background_node:
                # Also, if there is no background in the image,
                # bg.background_node == BG_NODE_NULL < 0 and this never occurs.
                data_p[i] = bg.background_label
            else:
                data_p[i] = counter
                # The background label is basically hardcoded to 0, so no need
                # to check that the new counter != bg.background_label
                counter += 1
        else:
            data_p[i] = data_p[forest_p[i]]
    return counter - 1


cdef void scanBG(DTYPE_t *data_p, DTYPE_t *forest_p, shape_info *shapeinfo,
                 bginfo *bg) nogil:
    """
    Settle all background pixels now and don't bother with them later.
    Since this only requires one linar sweep through the array, it is fast
    and it makes sense to do it separately.

    The purpose of this function is update of forest_p and bg parameter inplace.
    """
    cdef DTYPE_t i, bgval = bg.background_val, firstbg = shapeinfo.numels
    # We find the provisional label of the background, which is the index of
    # the first background pixel
    for i in range(shapeinfo.numels):
        if data_p[i] == bgval:
            firstbg = i
            bg.background_node = firstbg
            break

    # There is no background, therefore the first background element
    # is not defined.
    # Since BG_NODE_NULL < 0, this is enough to ensure
    # that resolve_labels doesn't worry about background.
    if bg.background_node == BG_NODE_NULL:
        return

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
                 bginfo *bg, DTYPE_t connectivity, DTYPE_t y, DTYPE_t z) nogil:
    """
    Perform forward scan on a 1D object, usually the first row of an image
    """
    if shapeinfo.numels == 0:
        return
    # Initialize the first row
    cdef DTYPE_t x, rindex, bgval = bg.background_val
    cdef DTYPE_t *DEX = shapeinfo.DEX
    rindex = shapeinfo.ravel_index(0, y, z, shapeinfo)

    for x in range(1, shapeinfo.x):
        rindex += 1
        # Handle the first row
        if data_p[rindex] == bgval:
            # Nothing to do if we are background
            continue

        join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ed])


cdef void scan2D(DTYPE_t *data_p, DTYPE_t *forest_p, shape_info *shapeinfo,
                 bginfo *bg, DTYPE_t connectivity, DTYPE_t z) nogil:
    """
    Perform forward scan on a 2D array.
    """
    if shapeinfo.numels == 0:
        return
    cdef DTYPE_t x, y, rindex, bgval = bg.background_val
    cdef DTYPE_t *DEX = shapeinfo.DEX
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
                 bginfo *bg, DTYPE_t connectivity) nogil:
    """
    Perform forward scan on a 3D array.

    """
    if shapeinfo.numels == 0:
        return
    cdef DTYPE_t x, y, z, rindex, bgval = bg.background_val
    cdef DTYPE_t *DEX = shapeinfo.DEX
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
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_em])
                if connectivity >= 3:
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_el])
        # END of x = max
        # END of y = 0

        # BEGINNING of y = ...
        for y in range(1, shapeinfo.y - 1):
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
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ec])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eg])
                    join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ei])
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
        rindex = shapeinfo.ravel_index(0, shapeinfo.y - 1, z, shapeinfo)
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
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ec])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_eg])
                join_trees_wrapper(data_p, forest_p, rindex, DEX[D_ei])
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
