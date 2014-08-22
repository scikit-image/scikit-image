#cython: cdivision=True
#cython: nonecheck=False
"""Cython implementation of Dijkstra's minimum cost path algorithm,
for use with data on a n-dimensional lattice.

Original author: Zachary Pincus
Inspired by code from Almar Klein
Later modifications by Almar Klein (Dec 2013)

License: BSD

Copyright 2009 Zachary Pincus

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import cython
import numpy as np
import heap

cimport numpy as cnp
cimport heap

OFFSET_D = np.int8
OFFSETS_INDEX_D = np.int16
EDGE_D = np.int8
INDEX_D = np.intp
FLOAT_D = np.float64


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_edge_map(shape):
    """Return an array with edge points/lines/planes/hyperplanes marked.

    Given a shape (of length n), return an edge_map array with a shape of
    original_shape + (n,), where, for each dimension, edge_map[...,dim] will
    have zeros at indices not along an edge in that dimension, -1s at indices
    along the lower boundary, and +1s on the upper boundary.

    This allows one to, given an nd index, calculate not only if the index is
    at the edge of the array, but if so, which edge(s) it lies along.

    """
    d = len(shape)
    edges = np.zeros(shape+(d,), order='F', dtype=EDGE_D)
    for i in range(d):
        slices = [slice(None)] * (d+1)
        slices[d] = i
        slices[i] = 0
        edges[tuple(slices)] = -1
        slices[i] = -1
        edges[tuple(slices)] = 1
    return edges


@cython.boundscheck(False)
@cython.wraparound(False)
def _offset_edge_map(shape, offsets):
    """Return an array with positions marked where offsets will step
    out of bounds.

    Given a shape (of length n) and a list of n-d offsets, return a two arrays
    of (n,) + shape: pos_edge_map and neg_edge_map.
    For each dimension xxx_edge_map[dim, ...] has zeros at indices at which
    none of the given offsets (in that dimension) of the given sign (positive
    or negative, respectively) will step out of bounds. If the value is
    nonzero, it gives the largest offset (in terms of absolute value) that
    will step out of bounds in that direction.

    An example will be explanatory:
    >>> offsets = [[-2,0], [1,1], [0,2]]
    >>> pos_edge_map, neg_edge_map = _offset_edge_map((4,4), offsets)
    >>> neg_edge_map[0]
    array([[-1, -1, -1, -1],
          [-2, -2, -2, -2],
          [ 0,  0,  0,  0],
          [ 0,  0,  0,  0]], dtype=int8)

    >>> pos_edge_map[1]
    array([[0, 0, 2, 1],
          [0, 0, 2, 1],
          [0, 0, 2, 1],
          [0, 0, 2, 1]], dtype=int8)

    """
    indices = np.indices(shape)  # indices.shape = (n,)+shape

    #get the distance from each index to the upper or lower edge in each dim
    pos_edges = (shape - indices.T).T
    neg_edges = -1 - indices
    # now set the distances to zero if none of the given offsets could reach
    offsets = np.asarray(offsets)
    maxes = offsets.max(axis=0)
    mins = offsets.min(axis=0)
    for pos, neg, mx, mn in zip(pos_edges, neg_edges, maxes, mins):
        pos[pos > mx] = 0
        neg[neg < mn] = 0
    return pos_edges.astype(EDGE_D), neg_edges.astype(EDGE_D)


@cython.boundscheck(False)
@cython.wraparound(False)
def make_offsets(d, fully_connected):
    """Make a list of offsets from a center point defining a n-dim
    neighborhood.

    Parameters
    ----------
    d : int
        dimension of the offsets to produce
    fully_connected : bool
        whether the neighborhood should be singly- of fully-connected

    Returns
    -------
    offsets : list of tuples of length `d`

    Examples
    --------

    The singly-connected 2-d neighborhood is four offsets:

    >>> make_offsets(2, False)
    [(-1,0), (1,0), (0,-1), (0,1)]

    While the fully-connected 2-d neighborhood is the full cartesian product
    of {-1, 0, 1} (less the origin (0,0)).

    """
    if fully_connected:
        mask = np.ones([3]*d, dtype=np.uint8)
        mask[tuple([1]*d)] = 0
    else:
        mask = np.zeros([3]*d, dtype=np.uint8)
        for i in range(d):
            indices = [1]*d
            indices[i] = (0, -1)
            mask[tuple(indices)] = 1
    offsets = []
    for indices, value in np.ndenumerate(mask):
        if value == 1:
            indices = np.array(indices) - 1
            offsets.append(indices)
    return offsets


@cython.boundscheck(True)
@cython.wraparound(True)
def _unravel_index_fortran(flat_indices, shape):
    """_unravel_index_fortran(flat_indices, shape)

    Given a flat index into an n-d fortran-strided array, return an
    index tuple.

    """
    strides = np.multiply.accumulate([1] + list(shape[:-1]))
    indices = [tuple((idx // strides) % shape) for idx in flat_indices]
    return indices


@cython.boundscheck(True)
@cython.wraparound(True)
def _ravel_index_fortran(indices, shape):
    """_ravel_index_fortran(flat_indices, shape)

    Given an index tuple into an n-d fortran-strided array, return a
    flat index.

    """
    strides = np.multiply.accumulate([1] + list(shape[:-1]))
    flat_indices = [np.sum(strides * idx) for idx in indices]
    return flat_indices


@cython.boundscheck(False)
@cython.wraparound(False)
def _normalize_indices(indices, shape):
    """_normalize_indices(indices, shape)

    Make all indices positive. If an index is out-of-bounds, return None.

    """
    new_indices = []
    for index in indices:
        if len(index) != len(shape):
            return None
        new_index = []
        for i, s in zip(index, shape):
            i = int(i)
            if i < 0:
                i = s + i
            if not (0 <= i < s):
                return None
            new_index.append(i)
        new_indices.append(new_index)
    return new_indices


@cython.boundscheck(True)
@cython.wraparound(True)
def _reverse(arr):
    """Reverse index an array safely, with bounds/wraparound checks on.
    """
    return arr[::-1]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MCP:
    """MCP(costs, offsets=None, fully_connected=True, sampling=None)

    A class for finding the minimum cost path through a given n-d costs array.

    Given an n-d costs array, this class can be used to find the minimum-cost
    path through that array from any set of points to any other set of points.
    Basic usage is to initialize the class and call find_costs() with a one
    or more starting indices (and an optional list of end indices). After
    that, call traceback() one or more times to find the path from any given
    end-position to the closest starting index. New paths through the same
    costs array can be found by calling find_costs() repeatedly.

    The cost of a path is calculated simply as the sum of the values of the
    `costs` array at each point on the path. The class MCP_Geometric, on the
    other hand, accounts for the fact that diagonal vs. axial moves are of
    different lengths, and weights the path cost accordingly.

    Array elements with infinite or negative costs will simply be ignored, as
    will paths whose cumulative cost overflows to infinite.

    Parameters
    ----------
    costs : ndarray
    offsets : iterable, optional
        A list of offset tuples: each offset specifies a valid move from a
        given n-d position.
        If not provided, offsets corresponding to a singly- or fully-connected
        n-d neighborhood will be constructed with make_offsets(), using the
        `fully_connected` parameter value.
    fully_connected : bool, optional
        If no `offsets` are provided, this determines the connectivity of the
        generated neighborhood. If true, the path may go along diagonals
        between elements of the `costs` array; otherwise only axial moves are
        permitted.
    sampling : tuple, optional
        For each dimension, specifies the distance between two cells/voxels.
        If not given or None, the distance is assumed unit.

    Attributes
    ----------
    offsets : ndarray
        Equivalent to the `offsets` provided to the constructor, or if none
        were so provided, the offsets created for the requested n-d
        neighborhood. These are useful for interpreting the `traceback` array
        returned by the find_costs() method.

    """

    def __init__(self, costs, offsets=None, fully_connected=True,
                 sampling=None):
        """__init__(costs, offsets=None, fully_connected=True, sampling=None)

        See class documentation.
        """
        costs = np.asarray(costs)
        if not np.can_cast(costs.dtype, FLOAT_D):
            raise TypeError('cannot cast costs array to ' + str(FLOAT_D))

        # Check sampling
        if sampling is None:
            sampling = np.array([1.0 for s in costs.shape], FLOAT_D)
        elif isinstance(sampling, (list, tuple)):
            sampling = np.array(sampling, FLOAT_D)
            if sampling.ndim != 1 or len(sampling) != costs.ndim:
                raise ValueError('Need one sampling element per dimension.')
        else:
            raise ValueError('Invalid type for sampling: %r.' % type(sampling))

        # We use flat, fortran-style indexing here (could use C-style,
        # but this is my code and I like fortran-style! Also, it's
        # faster when working with image arrays, which are often
        # already fortran-strided.)
        self.flat_costs = costs.astype(FLOAT_D).flatten('F')
        size = self.flat_costs.shape[0]
        self.flat_cumulative_costs = np.empty(size, dtype=FLOAT_D)
        self.dim = len(costs.shape)
        self.costs_shape = costs.shape
        self.costs_heap = heap.FastUpdateBinaryHeap(initial_capacity=128,
                                                    max_reference=size-1)

        # This array stores, for each point, the index into the offset
        # array (see below) that leads to that point from the
        # predecessor point.
        self.traceback_offsets = np.empty(size, dtype=OFFSETS_INDEX_D)

        # The offsets are a list of relative offsets from a central
        # point to each point in the relevant neighborhood. (e.g. (-1,
        # 0) might be a 2d offset).
        # These offsets are raveled to provide flat, 1d offsets that can be
        # used in the same way for flat indices to move to neighboring points.
        if offsets is None:
            offsets = make_offsets(self.dim, fully_connected)
        self.offsets = np.array(offsets, dtype=OFFSET_D)
        self.flat_offsets = np.array(
            _ravel_index_fortran(self.offsets, self.costs_shape),
            dtype=INDEX_D)

        # Instead of unraveling each index during the pathfinding algorithm, we
        # will use a pre-computed "edge map" that specifies for each dimension
        # whether a given index is on a lower or upper boundary (or none at
        # all). Flatten this map to get something that can be indexed as by the
        # same flat indices as elsewhere.
        # The edge map stores more than a boolean "on some edge" flag so as to
        # allow us to examine the non-out-of-bounds neighbors for a given edge
        # point while excluding the neighbors which are outside the array.
        pos, neg = _offset_edge_map(costs.shape, self.offsets)
        self.flat_pos_edge_map = pos.reshape((self.dim, size), order='F')
        self.flat_neg_edge_map = neg.reshape((self.dim, size), order='F')


        # The offset lengths are the distances traveled along each offset
        self.offset_lengths = np.sqrt(np.sum((sampling * self.offsets)**2,
                                      axis=1)).astype(FLOAT_D)
        self.dirty = 0
        self.use_start_cost = 1


    def _reset(self):
        """_reset()
        Clears paths found by find_costs().
        """

        cdef INDEX_T start

        self.costs_heap.reset()
        self.traceback_offsets[...] = -2  # -2 is not reached, -1 is start
        self.flat_cumulative_costs[...] = np.inf
        self.dirty = 0

        # Get starts and ends
        # We do not pass them in as arguments for backwards compat
        starts, ends = self._starts, self._ends

        # push each start point into the heap. Note that we use flat indexing!
        for start in _ravel_index_fortran(starts, self.costs_shape):
            self.traceback_offsets[start] = -1
            if self.use_start_cost:
                self.costs_heap.push_fast(self.flat_costs[start], start)
            else:
                self.costs_heap.push_fast(0, start)


    cdef FLOAT_T _travel_cost(self, FLOAT_T old_cost,
                              FLOAT_T new_cost, FLOAT_T offset_length):
        """ float _travel_cost(float old_cost, float new_cost,
                               float offset_length)
        The travel cost for going from the current node to the next.
        Default is simply the cost of the next node.
        """
        return new_cost


    cpdef int goal_reached(self, INDEX_T index, FLOAT_T cumcost):
        """ int goal_reached(int index, float cumcost)
        This method is called each iteration after popping an index
        from the heap, before examining the neighbours.

        This method can be overloaded to modify the behavior of the MCP
        algorithm. An example might be to stop the algorithm when a
        certain cumulative cost is reached, or when the front is a
        certain distance away from the seed point.

        This method should return 1 if the algorithm should not check
        the current point's neighbours and 2 if the algorithm is now
        done.
        """
        return 0


    cdef void _examine_neighbor(self, INDEX_T index, INDEX_T new_index,
                                FLOAT_T offset_length):
        """ _examine_neighbor(int index, int new_index, float offset_length)
        This method is called once for every pair of neighboring nodes,
        as soon as both nodes become frozen.
        """
        pass


    cdef void _update_node(self, INDEX_T index, INDEX_T new_index,
                           FLOAT_T offset_length):
        """ _update_node(int index, int new_index, float offset_length)
        This method is called when a node is updated.
        """
        pass


    def find_costs(self, starts, ends=None, find_all_ends=True,
                   max_coverage=1.0, max_cumulative_cost=None, max_cost=None):
        """
        Find the minimum-cost path from the given starting points.

        This method finds the minimum-cost path to the specified ending
        indices from any one of the specified starting indices. If no end
        positions are given, then the minimum-cost path to every position in
        the costs array will be found.

        Parameters
        ----------
        starts : iterable
            A list of n-d starting indices (where n is the dimension of the
            `costs` array). The minimum cost path to the closest/cheapest
            starting point will be found.
        ends : iterable, optional
            A list of n-d ending indices.
        find_all_ends : bool, optional
            If 'True' (default), the minimum-cost-path to every specified
            end-position will be found; otherwise the algorithm will stop when
            a a path is found to any end-position. (If no `ends` were
            specified, then this parameter has no effect.)

        Returns
        -------
        cumulative_costs : ndarray
            Same shape as the `costs` array; this array records the minimum
            cost path from the nearest/cheapest starting index to each index
            considered. (If `ends` were specified, not all elements in the
            array will necessarily be considered: positions not evaluated will
            have a cumulative cost of inf. If `find_all_ends` is 'False', only
            one of the specified end-positions will have a finite cumulative
            cost.)
        traceback : ndarray
            Same shape as the `costs` array; this array contains the offset to
            any given index from its predecessor index. The offset indices
            index into the `offsets` attribute, which is a array of n-d
            offsets. In the 2-d case, if offsets[traceback[x, y]] is (-1, -1),
            that means that the predecessor of [x, y] in the minimum cost path
            to some start position is [x+1, y+1]. Note that if the
            offset_index is -1, then the given index was not considered.

        """
        # basic variables to use for end-finding; also fix up the start and end
        # lists
        cdef BOOL_T use_ends = 0
        cdef INDEX_T num_ends
        cdef BOOL_T all_ends = find_all_ends
        cdef INDEX_T[:] flat_ends
        starts = _normalize_indices(starts, self.costs_shape)
        if starts is None:
            raise ValueError('start points must all be within the costs array')
        elif not starts:
            raise ValueError('no valid start points to start front' +
                             'propagation')
        if ends is not None:
            ends = _normalize_indices(ends, self.costs_shape)
            if ends is None:
                raise ValueError('end points must all be within '
                                 'the costs array')
            use_ends = 1
            num_ends = len(ends)
            flat_ends = np.array(_ravel_index_fortran(
                ends, self.costs_shape), dtype=INDEX_D)

        # Always perform a reset to (re)initialize our arrays and start
        # positions
        self._starts, self._ends = starts, ends
        self._reset()

        # Get shorter names for arrays
        cdef FLOAT_T[:] flat_costs = self.flat_costs
        cdef FLOAT_T[:] flat_cumulative_costs = self.flat_cumulative_costs
        cdef OFFSETS_INDEX_T[:] traceback_offsets = self.traceback_offsets
        cdef EDGE_T[:, :] flat_pos_edge_map = self.flat_pos_edge_map
        cdef EDGE_T[:, :] flat_neg_edge_map = self.flat_neg_edge_map
        cdef OFFSET_T[:, :] offsets = self.offsets
        cdef INDEX_T[:] flat_offsets = self.flat_offsets
        cdef FLOAT_T[:] offset_lengths = self.offset_lengths

        # Short names for other attributes
        cdef heap.FastUpdateBinaryHeap costs_heap = self.costs_heap
        cdef DIM_T dim = self.dim
        cdef int num_offsets = len(flat_offsets)

        # Variables used during front propagation
        cdef FLOAT_T cost, new_cost, cumcost, new_cumcost, offset_length
        cdef INDEX_T index, new_index
        cdef BOOL_T is_at_edge, use_offset
        cdef INDEX_T d, i, iter
        cdef OFFSET_T offset
        cdef EDGE_T pos_edge_val, neg_edge_val
        cdef int num_ends_found = 0
        cdef FLOAT_T inf = np.inf
        cdef int goal_reached

        cdef INDEX_T maxiter = int(max_coverage * flat_costs.size)

        for iter in range(maxiter):

            # This is rather like a while loop, except we are guaranteed to
            # exit, which is nice during developing to prevent eternal loops.

            # Find the point with the minimum cost in the heap. Once
            # popped, this point's minimum cost path has been found.
            if costs_heap.count == 0:
                # nothing in the heap: we've found paths to every
                # point in the array
                break

            # Get current cumulative cost and index from the heap
            cumcost = costs_heap.pop_fast()
            index = costs_heap._popped_ref

            # Record the cost we found to this point
            flat_cumulative_costs[index] = cumcost

            # Check if goal is reached
            goal_reached = self.goal_reached(index, cumcost)
            if goal_reached > 0:
                if goal_reached == 1:
                    continue  # Skip neighbours
                else:
                    break  # Done completely

            if use_ends:
                # If we're only tracing out a path to one or more
                # endpoints, check to see if this is an endpoint, and
                # if so, if we're done pathfinding.
                for i in range(num_ends):
                    if index == flat_ends[i]:
                        num_ends_found += 1
                        break
                if (num_ends_found and not all_ends) or \
                    num_ends_found == num_ends:
                    # if we've found one or all of the end points (as
                    # requested), stop searching
                    break

            # Look into the edge map to see if this point is at an
            # edge along any axis
            is_at_edge = 0
            for d in range(dim):
                if (flat_pos_edge_map[d, index] != 0 or
                    flat_neg_edge_map[d, index] != 0):
                    is_at_edge = 1
                    break

            # Now examine the points neighboring the given point
            for i in range(num_offsets):
                # First, if we're at some edge, scrutinize the offset
                # to ensure that it won't put us out-of-bounds. If,
                # for example, the edge_map at (x, y) is (-1, 0) --
                # though of course we use flat indexing below -- that
                # means that (x, y) is along the lower edge of the
                # array; thus offsets with -1 or more negative in the
                # x-dimension should not be used!
                use_offset = 1
                if is_at_edge:
                    for d in range(dim):
                        offset = offsets[i, d]
                        pos_edge_val = flat_pos_edge_map[d, index]
                        neg_edge_val = flat_neg_edge_map[d, index]
                        if (pos_edge_val > 0 and offset >= pos_edge_val) or \
                           (neg_edge_val < 0 and offset <= neg_edge_val):
                            # the offset puts us out of bounds...
                            use_offset = 0
                            break
                # If not at an edge, or the specific offset doesn't
                # push over the edge, then we go on.
                if not use_offset:
                    continue

                # using the flat offsets, calculate the new flat index
                new_index = index + flat_offsets[i]

                # Get offset length
                offset_length = offset_lengths[i]

                # If we have already found the best path here then
                # ignore this point
                if flat_cumulative_costs[new_index] != inf:
                    # Give subclass the oportunity to examine these two nodes
                    # Note that only when both nodes are "frozen" their
                    # cumulative cost is set. By doing the check here, each
                    # pair of nodes is checked exactly once.
                    self._examine_neighbor(index, new_index, offset_length)
                    continue

                # Get cost and new cost
                cost = flat_costs[index]
                new_cost = flat_costs[new_index]

                # If the cost at this point is negative or infinite, ignore it
                if new_cost < 0 or new_cost == inf:
                    continue

                # Calculate new cumulative cost
                new_cumcost = cumcost + self._travel_cost(cost, new_cost,
                                                          offset_length)

                # Now we ask the heap to append or update the cost to
                # this new point, but only if that point isn't already
                # in the heap, or it is but the new cost is lower.
                # don't push infs into the heap though!
                if new_cumcost != inf:
                    costs_heap.push_if_lower_fast(new_cumcost, new_index)
                    # If we did perform an append or update, we should
                    # record the offset from the predecessor to this new
                    # point
                    if costs_heap._pushed:
                        traceback_offsets[new_index] = i
                        self._update_node(index, new_index, offset_length)


        # Un-flatten the costs and traceback arrays for human consumption.
        cumulative_costs = np.asarray(flat_cumulative_costs)
        cumulative_costs = cumulative_costs.reshape(self.costs_shape,
                                                    order='F')
        traceback = np.asarray(traceback_offsets)
        traceback = traceback.reshape(self.costs_shape, order='F')
        self.dirty = 1
        return cumulative_costs, traceback


    def traceback(self, end):
        """traceback(end)

        Trace a minimum cost path through the pre-calculated traceback array.

        This convenience function reconstructs the the minimum cost path to a
        given end position from one of the starting indices provided to
        find_costs(), which must have been called previously. This function
        can be called as many times as desired after find_costs() has been
        run.

        Parameters
        ----------
        end : iterable
            An n-d index into the `costs` array.

        Returns
        -------
        traceback : list of n-d tuples
            A list of indices into the `costs` array, starting with one of
            the start positions passed to find_costs(), and ending with the
            given `end` index. These indices specify the minimum-cost path
            from any given start index to the `end` index. (The total cost
            of that path can be read out from the `cumulative_costs` array
            returned by find_costs().)
        """
        if not self.dirty:
            raise Exception('find_costs() must be run before traceback()')
        ends = _normalize_indices([end], self.costs_shape)
        if ends is None:
            raise ValueError('the specified end point must be '
                             'within the costs array')
        traceback = [tuple(ends[0])]

        cdef INDEX_T flat_position =\
             _ravel_index_fortran(ends, self.costs_shape)[0]
        if self.flat_cumulative_costs[flat_position] == np.inf:
            raise ValueError('no minimum-cost path was found '
                             'to the specified end point')

        # Short names for arrays
        cdef OFFSETS_INDEX_T [:] traceback_offsets = self.traceback_offsets
        cdef OFFSET_T [:,:] offsets = self.offsets
        cdef INDEX_T [:] flat_offsets = self.flat_offsets
        # New array
        cdef INDEX_T [:] position = np.array(ends[0], dtype=INDEX_D)

        cdef OFFSETS_INDEX_T offset
        cdef DIM_T d
        cdef DIM_T dim = self.dim
        while 1:
            offset = traceback_offsets[flat_position]
            if offset == -1:
                # At a point where we can go no further: probably a start point
                break
            flat_position -= flat_offsets[offset]
            for d in range(dim):
                position[d] -= offsets[offset, d]
            traceback.append(tuple(position))
        return _reverse(traceback)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MCP_Geometric(MCP):
    """MCP_Geometric(costs, offsets=None, fully_connected=True)

    Find distance-weighted minimum cost paths through an n-d costs array.

    See the documentation for MCP for full details. This class differs from
    MCP in that the cost of a path is not simply the sum of the costs along
    that path.

    This class instead assumes that the costs array contains at each position
    the "cost" of a unit distance of travel through that position. For
    example, a move (in 2-d) from (1, 1) to (1, 2) is assumed to originate in
    the center of the pixel (1, 1) and terminate in the center of (1, 2). The
    entire move is of distance 1, half through (1, 1) and half through (1, 2);
    thus the cost of that move is `(1/2)*costs[1,1] + (1/2)*costs[1,2]`.

    On the other hand, a move from (1, 1) to (2, 2) is along the diagonal and
    is sqrt(2) in length. Half of this move is within the pixel (1, 1) and the
    other half in (2, 2), so the cost of this move is calculated as
    `(sqrt(2)/2)*costs[1,1] + (sqrt(2)/2)*costs[2,2]`.

    These calculations don't make a lot of sense with offsets of magnitude
    greater than 1. Use the `sampling` argument in order to deal with
    anisotropic data.
    """

    def __init__(self, costs, offsets=None, fully_connected=True,
                 sampling=None):
        """__init__(costs, offsets=None, fully_connected=True, sampling=None)

        See class documentation.
        """
        MCP.__init__(self, costs, offsets, fully_connected, sampling)
        if np.absolute(self.offsets).max() > 1:
            raise ValueError('all offset components must be 0, 1, or -1')
        self.use_start_cost = 0

    cdef FLOAT_T _travel_cost(self, FLOAT_T old_cost, FLOAT_T new_cost,
                              FLOAT_T offset_length):
        return offset_length * 0.5 * (old_cost + new_cost)



@cython.boundscheck(True)
@cython.wraparound(True)
cdef class MCP_Connect(MCP):
    """MCP_Connect(costs, offsets=None, fully_connected=True)

    Connect source points using the distance-weighted minimum cost function.

    A front is grown from each seed point simultaneously, while the
    origin of the front is tracked as well. When two fronts meet,
    create_connection() is called. This method must be overloaded to
    deal with the found edges in a way that is appropriate for the
    application.
    """

    cdef INDEX_T [:] flat_idmap


    def __init__(self, costs, offsets=None, fully_connected=True,
                 sampling=None):
        MCP.__init__(self, costs, offsets, fully_connected, sampling)

        # Create id map to keep track of origin of nodes
        self.flat_idmap = np.zeros(self.costs_shape, INDEX_D).ravel('F')


    def _reset(self):
        """ Reset the id map.
        """
        cdef INDEX_T start

        MCP._reset(self)
        starts, ends = self._starts, self._ends

        # Reset idmap
        self.flat_idmap[...] = -1
        id = 0
        for start in _ravel_index_fortran(starts, self.costs_shape):
            self.flat_idmap[start] = id
            id += 1


    cdef FLOAT_T _travel_cost(self, FLOAT_T old_cost, FLOAT_T new_cost,
                              FLOAT_T offset_length):
        """ Equivalent to MCP_Geometric.
        """
        return offset_length * 0.5 * (old_cost + new_cost)


    cdef void _examine_neighbor(self, INDEX_T index, INDEX_T new_index,
                                FLOAT_T offset_length):
        """ Check whether two fronts are meeting. If so, the flat_traceback
        is obtained and a connection is created.
        """

        # Short names
        cdef INDEX_T [:] flat_idmap = self.flat_idmap
        cdef FLOAT_T [:] flat_cumulative_costs = self.flat_cumulative_costs

        # Get ids
        cdef INDEX_T id1 = flat_idmap[index]
        cdef INDEX_T id2 = flat_idmap[new_index]

        if id2 < 0 or id1 < 0:
            pass
        elif id2 != id1:
            # We reached the 'front' of another seed point!
            # Get position/coordinates
            pos1, pos2 = _unravel_index_fortran([index, new_index],
                                                self.costs_shape)
            # Also get the costs, so we can keep the path with the least cost
            cost1 = flat_cumulative_costs[index]
            cost2 = flat_cumulative_costs[new_index]
            # Create connection
            self.create_connection(id1, id2, pos1, pos2, cost1, cost2)


    def create_connection(self, id1, id2, tb1, tb2, cost1, cost2):
        """ create_connection id1, id2, pos1, pos2, cost1, cost2)

        Overload this method to keep track of the connections that are
        found during MCP processing. Note that a connection with the
        same ids can be found multiple times (but with different
        positions and costs).

        At the time that this method is called, both points are "frozen"
        and will not be visited again by the MCP algorithm.

        Parameters
        ----------
        id1 : int
            The seed point id where the first neighbor originated from.
        id2 : int
            The seed point id where the second neighbor originated from.
        pos1 : tuple
            The index of of the first neighbour in the connection.
        pos2 : tuple
            The index of of the second neighbour in the connection.
        cost1 : float
            The cumulative cost at `pos1`.
        cost2 : float
            The cumulative costs at `pos2`.
        """
        pass


    cdef void _update_node(self, INDEX_T index, INDEX_T new_index,
                           FLOAT_T offset_length):
        """ Keep track of the id map so that we know which seed point
        a certain front originates from.
        """
        self.flat_idmap[new_index] = self.flat_idmap[index]



@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MCP_Flexible(MCP):
    """MCP_Flexible(costs, offsets=None, fully_connected=True)

    Find minimum cost paths through an N-d costs array.

    See the documentation for MCP for full details. This class differs from
    MCP in that several methods can be overloaded (from pure Python) to
    modify the behavior of the algorithm and/or create custom algorithms
    based on MCP. Note that goal_reached can also be overloaded in the
    MCP class.

    """

    def travel_cost(self, FLOAT_T old_cost, FLOAT_T new_cost,
                    FLOAT_T offset_length):
        """ travel_cost(old_cost, new_cost, offset_length)
        This method calculates the travel cost for going from the
        current node to the next. The default implementation returns
        new_cost. Overload this method to adapt the behaviour of the
        algorithm.
        """
        return new_cost


    def examine_neighbor(self, INDEX_T index, INDEX_T new_index,
                         FLOAT_T offset_length):
        """ examine_neighbor(index, new_index, offset_length)
        This method is called once for every pair of neighboring nodes,
        as soon as both nodes are frozen.

        This method can be overloaded to obtain information about
        neightboring nodes, and/or to modify the behavior of the MCP
        algorithm. One example is the MCP_Connect class, which checks
        for meeting fronts using this hook.
        """
        pass


    def update_node(self, INDEX_T index, INDEX_T new_index,
                    FLOAT_T offset_length):
        """ update_node(index, new_index, offset_length)
        This method is called when a node is updated, right after
        new_index is pushed onto the heap and the traceback map is
        updated.

        This method can be overloaded to keep track of other arrays
        that are used by a specific implementation of the algorithm.
        For instance the MCP_Connect class uses it to update an id map.
        """
        pass


    cdef FLOAT_T _travel_cost(self, FLOAT_T old_cost, FLOAT_T new_cost,
                              FLOAT_T offset_length):
        return self.travel_cost(old_cost, new_cost, offset_length)


    cdef void _examine_neighbor(self, INDEX_T index, INDEX_T new_index,
                                FLOAT_T offset_length):
        self.examine_neighbor(index, new_index, offset_length)


    cdef void _update_node(self, INDEX_T index, INDEX_T new_index,
                           FLOAT_T offset_length):
        self.update_node(index, new_index, offset_length)
