# -*- python -*-

import numpy as numpy
cimport numpy as numpy
cimport cython

@cython.boundscheck(False)
def trace_path(numpy.ndarray[numpy.float32_t, ndim=2] costs not None,
               start, ends, diagonal_steps=True):
  """Find the lowest-cost path from the start point to each given end point.

  Inputs: 'costs' array; 'start' (x, y) pair; list of 'ends' (x, y) pairs,
    and optional 'diagonal_steps' boolean flag.

  Costs are given by the input array: a move onto any given position in the
  costs array adds that cost to the path. Paths may be constrained to
  vertical and horizontal moves only by passing False for the diagonal_steps
  parameter. The costs must be non-negative!

  The array of cumulative costs from the starting point, and a list of paths
  from the start to each end point are returned.

  Paths are found by (more or less) breadth-first search outward from the
  starting point: each time a lower-cost route to a given pixel is found, that
  pixel is marked "active"; the neighbors of all active pixels are then
  examined to see if their costs can be lowered as well. This continues until
  no pixels are marked active.
  """
  if costs.min() < 0:
    raise ValueError("All costs must be non-negative.")
  try:
    a, b = start
  except:
    raise ValueError("The start point must be an (x, y) pair")
  if not (0 <= a < costs.shape[0]  and 0 <= b < costs.shape[1]):
    raise ValueError("The start point must fall within the array")
  for end in ends:
    try:
      a, b = end
    except:
      raise ValueError("All end points must be (x, y) pairs")
    if not (0 <= a < costs.shape[0]  and 0 <= b < costs.shape[1]):
      raise ValueError("The end points must fall within the array")

  cdef numpy.ndarray[numpy.float32_t, ndim=2] cumulative_costs = \
       numpy.empty_like(costs)

  cumulative_costs.fill(numpy.inf)
  cumulative_costs[start] = 0
  costs_shape = (costs.shape[0], costs.shape[1])
  cdef numpy.ndarray[numpy.uint8_t, ndim=2] active_nodes = \
       numpy.zeros(costs_shape, dtype=numpy.uint8)

  active_nodes[start] = 1
  cdef numpy.ndarray[numpy.uint8_t, ndim=2] parent_nodes = \
       numpy.empty(costs_shape, dtype=numpy.uint8)

  parent_nodes.fill(255)
  cdef numpy.ndarray[numpy.int8_t, ndim=2] offsets
  if diagonal_steps:
      offsets = numpy.array([[-1, -1],
                             [-1,  0],
                             [-1,  1],
                             [ 0, -1],
                             [ 0,  1],
                             [ 1, -1],
                             [ 1,  0],
                             [ 1,  1]], dtype=numpy.int8)
  else:
      offsets = numpy.array([[-1, 0],
                             [0, -1],
                             [0, 1],
                             [1, 0]], dtype=numpy.int8)

  cdef Py_ssize_t x, y, ox, oy, xo, yo, i
  cdef Py_ssize_t a_xmax, a_xmin, a_ymax, a_ymin, tmp_xmax, \
                  tmp_xmin, tmp_ymax, tmp_ymin
  cdef unsigned int xmax, ymax, active, num_steps
  xmax = costs.shape[0] - 1
  ymax = costs.shape[1] - 1
  num_steps = 0
  tmp_xmax = tmp_xmin = start[0]
  tmp_ymax = tmp_ymin = start[1]
  cdef float current_cost, current_cumulative_cost, cumulative_cost, new_cost

  while True:
      active = 0
      # iterate over array
      for x in range(0, xmax + 1):
          for y in range(0, ymax + 1):
              if active_nodes[x, y]:
                  active_nodes[x, y] = 0
                  active = 1
                  current_cumulative_cost = cumulative_costs[x, y]

                  # iterate over offsets
                  for i in range(8):
                      ox = offsets[i, 0]
                      oy = offsets[i, 1]
                      xo = x + ox
                      yo = y + oy
                      if xo < 0 or xo > xmax or yo < 0 or yo > ymax:
                          continue

                      current_cost = costs[xo, yo]
                      new_cost = current_cost + current_cumulative_cost

                      # if a cheaper path to a given point is found,
                      # activate that point
                      if cumulative_costs[xo, yo] > new_cost:
                          cumulative_costs[xo, yo] = new_cost
                          parent_nodes[xo, yo] = i
                          active_nodes[xo, yo] = 1

      if not active:
        break

  cdef unsigned int startx, starty
  startx = start[0]
  starty = start[1]
  return_paths = []
  # Trace the paths from the endpoints to the start
  for end in ends:
      path = None
      x = end[0]
      y = end[1]
      if cumulative_costs[x, y] != numpy.inf:
          path = [(x, y)]
          while not (x == startx and y == starty):
              i = parent_nodes[x, y]
              ox = offsets[i, 0]
              oy = offsets[i, 1]
              x -= ox
              y -= oy
              path.append((x, y))
          path.reverse()
      return_paths.append(path)
  return cumulative_costs, return_paths
