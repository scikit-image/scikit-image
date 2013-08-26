import numpy as np
from cython cimport view
cimport numpy as cnp

DEF BETA = (1.0/5)
DEF EPSILON = 0.05
DEF AFFINITY = 60.0
DEF GLOBAL_RELABEL_INTERVAL = 8

cdef class Structure:
    DIM_2D_4 = 0
    DIM_2D_8 = 1
    DIM_3D_6 = 2
    DIM_3D_26 = 3

    cdef:
        list indices
        int cardinality

    def __init__(self, shape, case=DIM_2D_4):
        if case == Structure.DIM_2D_4:
            self.indices = [
                (-1, 0), #N
                (0, -1), #W
                (1, 0), #S
                (0, 1), #E
            ]
            self.cardinality = 4

        if case == Structure.DIM_2D_8:
            self.indices = [
                (-1, 0), #N
                (-1, -1), #NW
                (0, -1), #W
                (1, -1), #SW
                (1, 0), #S
                (1, 1), #SE
                (0, 1), #E
                (-1, 1), #NE
            ]
            self.cardinality = 8

cdef weight(Node node1, Node node2):
    return 1.0 / (BETA * np.sqrt(np.sum((node1.value - node2.value)**2)) + EPSILON)

cdef class Node:
    cdef:
        int height
        double excess
        tuple coord
        double[:] edges
        list neighbours
        object value #is either double (Grayscale) or ndarray (RGB)

    def __init__(self):
        pass

cdef class GraphCut:
    cdef:
        tuple shape
        readonly int size
        list nodes
        list active
        Structure structure

    def __init__(self, cnp.ndarray img, cnp.ndarray src, cnp.ndarray sink,
                 case=Structure.DIM_2D_4, double affinity=AFFINITY):
        cdef:
            int x, y
            int i
            tuple index
            x2, y2
            Node node, neighbour

        if affinity == None:
            affinity = AFFINITY

        self.structure = Structure(case)

        if case == Structure.DIM_2D_4:
            self.shape = (img.shape[0], img.shape[1])

            self.size = self.shape[1]*self.shape[0]
            self.nodes = [Node() for i in range(self.size)]

            for i in range(self.size):
                y = i/self.shape[1]
                x = i%self.shape[1]

                node = self.nodes[i]
                node.coord = (y, x)

                node.edges = np.zeros((self.structure.cardinality))
                node.neighbours = [None]*self.structure.cardinality

                for i in range(self.structure.cardinality):
                    index = self.structure.indices[i]

                    y2 = y + index[0]
                    x2 = x + index[1]

                    if y2 < 0 or\
                       x2 < 0 or\
                       y2 > self.shape[0] - 1 or\
                       x2 > self.shape[1] - 1:
                        node.neighbours[i] = None

                    else:
                        neighbour = self.nodes[y2*self.shape[1] + x2]

                        node.neighbours[i] = neighbour

        self.active = []

        for node in self.nodes:
            y, x = node.coord
            node.value = img[y][x]
            node.height = 0
            node.excess = src[y][x] - sink[y][x]

            if node.excess > 0:
                self.active.append(node)

            i += 1

        for node in self.nodes:
            for i in range(self.structure.cardinality):
                neighbour = node.neighbours[i]

                if neighbour == None:
                    continue

                node.edges[i] = affinity * weight(node, neighbour)

    def relabel(self):
        cdef:
            Node node, neighbour
            int i
            int min_height

        for node in self.active:
            min_height = self.size

            for i in range(self.structure.cardinality):
                neighbour = node.neighbours[i]

                if neighbour == None:
                    continue;

                if node.edges[i] > 0:
                    min_height = min(min_height, neighbour.height + 1)

            node.height = min_height

    def push(self):
        cdef:
            list active_new
            Node node, neighbour
            int i
            double flow
            int reverse_index

        active_new = []

        for node in self.active:
            for i in range(self.structure.cardinality):
                neighbour = node.neighbours[i]

                if neighbour == None:
                    continue;

                flow = min(node.excess, node.edges[i])

                if flow > 0 and node.height > neighbour.height:
                    node.excess -= flow
                    neighbour.excess += flow

                    node.edges[i] -= flow
                    reverse_index = (i + self.structure.cardinality/2) %\
                                    self.structure.cardinality
                    neighbour.edges[reverse_index] += flow

                    if node.excess > 0:
                        active_new.append(node)

                    if neighbour.excess > 0:
                        active_new.append(neighbour)

        self.active = active_new

    def global_relabel(self):
        cdef:
            Node node, neighbour
            list fifo_from, fifo_to
            int distance
            int reverse_index

        self.active = []

        fifo_from = []
        fifo_to = None

        for node in self.nodes:
            if node.excess < 0:
                fifo_from.append(node)
                node.height = 0
            else:
                node.height = self.size

        distance = 1

        while len(fifo_from) > 0:
            fifo_to = []

            for node in fifo_from:
                for i in range(self.structure.cardinality):
                    neighbour = node.neighbours[i]

                    if neighbour == None:
                        continue;

                    neighbour = node.neighbours[i]

                    reverse_index = (i + self.structure.cardinality/2) %\
                                    self.structure.cardinality
                    if neighbour.edges[reverse_index] > 0 and neighbour.height == self.size:
                        neighbour.height = distance

                        fifo_to.append(neighbour)

                        if neighbour.excess > 0:
                            self.active.append(neighbour)

            distance += 1

            fifo_from = fifo_to

    def getHeight(self):
        cdef:
            Node node

        height = np.empty(self.shape, np.int)

        for node in self.nodes:
            height[node.coord] = node.height

        return height

    cdef cut(self, int global_relabel_interval=GLOBAL_RELABEL_INTERVAL):
        cdef:
            int i

        self.global_relabel()

        i = 0
        while len(self.active) > 0:
            self.push()

            if i % global_relabel_interval == 0:
                self.global_relabel()
            else:
                self.relabel()

            i += 1

def graphcut(img, src, sink, dim=Structure.DIM_2D_4,
             global_relabel_interval=GLOBAL_RELABEL_INTERVAL,
             affinity=AFFINITY):
    """
    Performs a graph-cut on an image to return a globally optimal segmentation.

    Parameters
    ----------
    img : array
      The image to segment

    src : array
      The foreground negative log-liklihood

    sink : array
      The background negative log-liklihood

    global_relabel_interval : int
      The number of standard push/relabel iterations between a global
      relabel to calculate the exact distance to sink for each element in
      the height array

    affinity
      The data term, neighboorhood term weighting. Lower values give preference
      to the data term.

    Returns
    -------
    cut : ndarray
      A binary array of the optimal segmentation

    """

    gc = GraphCut(img, src, sink, dim, affinity)
    gc.cut(global_relabel_interval)

    shape = (img.shape[0], img.shape[1])
    cut = np.zeros(shape)
    height = np.array(gc.getHeight()).reshape(shape)

    cut[height == gc.size] = 1

    return cut
