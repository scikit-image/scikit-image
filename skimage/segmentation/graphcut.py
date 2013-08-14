import numpy as np

BETA = (1.0/10)
EPSILON = 0.05
LAMBA = 60

class Structure:
    DIM_2D = 0
    DIM_3D = 1

    CONNECTED_4 = 0
    CONNECTED_8 = 1

    def __init__(self, shape, connected=CONNECTED_4):
        if connected == Structure.CONNECTED_4:
            self.indices = [
                (-1, 0), #up
                (0, -1), #left
                (1, 0), #down
                (0, 1), #right
            ]
            self.cardinality = 4

def weight(node1, node2):
    return LAMBA * (1.0 / (BETA * np.sqrt(np.sum((node1.value - node2.value)**2)) + EPSILON))

class Graphcut:
    class Node:
        def __init__(self):
            self.height = 0
            self.excess = 0
            self.edges = None
            self.neighbours = None
            self.value = None
            self.coord = None

    def __init__(self, img, src, sink, dim=Structure.DIM_2D, connected=Structure.CONNECTED_4):
        self.structure = Structure(dim, connected)
        self.shape = img.shape

        if dim == Structure.DIM_2D:
            self.size = self.shape[1]*self.shape[0]
            self.nodes = [Graphcut.Node() for i in range(self.size)]

            for i in range(self.size):
                y = i/self.shape[1]
                x = i%self.shape[1]

                node = self.nodes[i]
                node.coord = (y, x)

                if connected == Structure.CONNECTED_4:
                    node.edges = [None]*4
                    node.neighbours = [None]*4

                for i in range(self.structure.cardinality):
                    index = self.structure.indices[i]

                    y2 = y + index[0]
                    x2 = x + index[1]
                    if y2 < 0 or\
                       x2 < 0 or\
                       y2 > self.shape[0] - 1 or\
                       x2 > self.shape[1] - 1:
                        continue

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

        for node in self.nodes:
            for i in range(len(node.neighbours)):
                if node.neighbours[i] == None:
                    continue

                node.edges[i] = weight(node, neighbour)

    def relabel(self):
        for node in self.active:
            min_height = self.size

            for i in range(len(node.neighbours)):
                if node.neighbours[i] == None:
                    continue;

                if node.edges[i] > 0:
                    min_height = min(min_height, node.neighbours[i].height + 1)

            node.height = min_height

    def push(self):
        active_new = []

        for node in self.active:
            for i in range(len(node.neighbours)):
                if node.neighbours[i] == None:
                    continue;

                flow = min(node.excess, node.edges[i])

                neighbour = node.neighbours[i]
                if flow > 0 and node.height > neighbour.height:
                    node.excess -= flow
                    neighbour.excess += flow

                    node.edges[i] -= flow
                    neighbour.edges[(i+(4/2)) % 4] -= flow

                    if node.excess > 0:
                        active_new.append(node)

                    if neighbour.excess > 0:
                        active_new.append(neighbour)

        self.active = active_new

    def global_relabel(self):
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
            print len(fifo_from), distance
            fifo_to = []

            for node in fifo_from:
                for i in range(len(node.neighbours)):
                    if node.neighbours[i] == None:
                        continue;

                    neighbour = node.neighbours[i]

                    if neighbour.edges[(i+(4/2)) % 4] > 0 and neighbour.height == self.size:

                        if neighbour in fifo_to:
                            print 'wtf'

                        neighbour.height = distance

                        fifo_to.append(neighbour)

                        if neighbour.excess > 0:
                            self.active.append(neighbour)

            distance = distance + 1

            fifo_from = fifo_to


import numpy as np
import Image
from skimage import data_dir

src = np.load(data_dir+'/'+'trolls_fg.npy')
sink = np.load(data_dir+'/'+'/trolls_bg.npy')
data = Image.open(data_dir+'/'+'trolls_small.png')
if data.mode != 'RGBA':
    data = data.convert('RGBA')

img = np.array(data)

gc = Graphcut(img, src, sink)

#left = np.empty((img.shape[0], img.shape[1]))
#excess = np.empty((img.shape[0], img.shape[1]))
height = np.empty((img.shape[0], img.shape[1]))

for node in gc.nodes:
    height[node.coord] = node.height
#    left[node.coord] = node.edges[0]
#    excess[node.coord] = node.excess

import matplotlib.pyplot as plt
plt.imshow(height, interpolation='nearest', vmin=0)
plt.show()

gc.global_relabel()

i = 0
while len(gc.active) > 0:
    print len(gc.active)
    gc.push()

    if i % 10 == 0:
        gc.global_relabel()
    else:
        gc.relabel()

    i += 1

True