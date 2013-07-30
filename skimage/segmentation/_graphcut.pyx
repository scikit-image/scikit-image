#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np

cimport cython
cimport numpy as cnp

BETA = (1.0/10)
EPSILON = 0.05
LAMBA = 60

cdef cnp.ndarray img

cdef class GraphCut:
    cdef:
        readonly double[:] excess, up, down, left, right
        Py_ssize_t[:] height
        int w, h
        int size
        readonly int max_label

        dict active
        list active_add
        list active_remove
        int i

    def __init__(self, img, src, sink, weight_lambda=None):
        if weight_lambda == None:
            weight_lambda = LAMBA

        self.excess = (sink - src).astype(np.double).ravel()
        self.height = np.zeros((600, 800), np.int).ravel()

        self.w = img.shape[1]
        self.h = img.shape[0]
        self.size = img.shape[0]*img.shape[1]
        self.max_label = self.size

        _down = img.copy()
        _up = img.copy()
        _right = img.copy()
        _left = img.copy()

        _down[:(self.h-1), :] = np.diff(img, axis=0)**2
        _up[1:, :] = np.diff(img, axis=0)**2
        _right[:,:(self.w-1)] = np.diff(img, axis=1)**2
        _left[:,1:] = np.diff(img, axis=1)**2

        _up = weight_lambda*(1.0 / (BETA * np.sqrt(np.sum(_up, 2)) + EPSILON))
        _down = weight_lambda*(1.0 / (BETA * np.sqrt(np.sum(_down, 2)) + EPSILON))
        _left = weight_lambda*(1.0 / (BETA * np.sqrt(np.sum(_left, 2)) + EPSILON))
        _right = weight_lambda*(1.0 / (BETA * np.sqrt(np.sum(_right, 2)) + EPSILON))

        _up[0] = 0
        _down[-1] = 0
        _left[:, 0] = 0
        _right[:, -1] = 0

        self.up = _up.ravel()
        self.down = _down.ravel()
        self.left = _left.ravel()
        self.right = _right.ravel()

        self.active = {}

        for i in range(self.size):
            if self.excess[i] > 0:
                self.active[i] = None

    cdef int index_up(self, int i):
        return i - self.w
    cdef int index_down(self, int i):
        return i + self.w
    cdef int index_left(self, int i):
        return i - 1
    cdef int index_right(self, int i):
        return i + 1

    cdef push(self):
        cdef:
            int i, x, y
            double flow

        active_remove = []
        active_add = []

        for i in self.active:
            flow = min(self.excess[i], self.up[i])
            if flow > 0 and self.height[i] > self.height[self.index_up(i)]:
                self.excess[i] -= flow
                self.excess[self.index_up(i)] += flow

                self.up[i] -= flow
                self.down[self.index_up(i)] += flow

                if self.excess[i] == 0:
                    active_remove.append(i)

                if self.excess[self.index_up(i)] > 0:
                    active_add.append(self.index_up(i))

            flow = min(self.excess[i], self.down[i])
            if flow > 0 and self.height[i] > self.height[self.index_down(i)]:
                self.excess[i] -= flow
                self.excess[self.index_down(i)] += flow

                self.down[i] -= flow
                self.up[self.index_down(i)] += flow

                if self.excess[i] == 0:
                    active_remove.append(i)

                if self.excess[self.index_down(i)] > 0:
                    active_add.append(self.index_down(i))

            flow = min(self.excess[i], self.left[i])
            if flow > 0 and self.height[i] > self.height[self.index_left(i)]:
                self.excess[i] -= flow
                self.excess[self.index_left(i)] += flow

                self.left[i] -= flow
                self.right[self.index_left(i)] += flow

                if self.excess[i] == 0:
                    active_remove.append(i)

                if self.excess[self.index_left(i)] > 0:
                    active_add.append(self.index_left(i))

            flow = min(self.excess[i], self.right[i])
            if flow > 0 and self.height[i] > self.height[self.index_right(i)]:
                self.excess[i] -= flow
                self.excess[self.index_right(i)] += flow

                self.right[i] -= flow
                self.left[self.index_right(i)] += flow

                if self.excess[i] == 0:
                    active_remove.append(i)

                if self.excess[self.index_right(i)] > 0:
                    active_add.append(self.index_right(i))

        for i in active_remove:
            del self.active[i]

        for i in active_add:
            self.active[i] = None

    cdef relabel(self):
        cdef:
            double flow
            int min_height
            int i

        for i in self.active:
            min_height = self.size

            if self.up[i] > 0:
                min_height = min(min_height, self.height[self.index_up(i)] + 1)
            if self.down[i] > 0:
                min_height = min(min_height, self.height[self.index_down(i)] + 1)
            if self.left[i] > 0:
                min_height = min(min_height, self.height[self.index_left(i)] + 1)
            if self.right[i] > 0:
                min_height = min(min_height, self.height[self.index_right(i)] + 1)

            self.height[i] = min_height

    cdef global_relabel(self):
        cdef:
            int i
            list fifo_from = []
            list fifo_to
            list fifo_temp
            int distance

        self.active.clear()

        for i in range(self.size):
            #all nodes with excess < 0 have an edge to the sink still, so their distance should be 0
            if self.excess[i] < 0:
                fifo_from.append(i)
                self.height[i] = 0
            #nodes with excess = 0, dont have an edge to the sink need to be included in the BFS
            #unreachable nodes should be labeled as max_label after BFS, so label themm now,
            #if they arent reachedm their label=max_label is retained...
            else:
                self.height[i] = self.max_label

        distance = 1

        while len(fifo_from) > 0:
            fifo_to = []

            for i in fifo_from:
                if i/self.w > 0:
                    if  self.down[self.index_up(i)] > 0 and self.height[self.index_up(i)] == self.max_label:
                        self.height[self.index_up(i)] = distance

                        fifo_to.append(self.index_up(i))

                        if self.excess[self.index_up(i)] > 0:
                            self.active[self.index_up(i)] = None

                if i/self.w < self.h-1:
                    if  self.up[self.index_down(i)] > 0 and self.height[self.index_down(i)] == self.max_label:
                        self.height[self.index_down(i)] = distance

                        fifo_to.append(self.index_down(i))

                        if self.excess[self.index_down(i)] > 0:
                            self.active[self.index_down(i)] = None

                if i%self.w > 0:
                    if  self.right[self.index_left(i)] > 0 and self.height[self.index_left(i)] == self.max_label:
                        self.height[self.index_left(i)] = distance

                        fifo_to.append(self.index_left(i))

                        if self.excess[self.index_left(i)] > 0:
                            self.active[self.index_left(i)] = None

                if i%self.w < self.w-1:
                    if  self.left[self.index_right(i)] > 0 and self.height[self.index_right(i)] == self.max_label:
                        self.height[self.index_right(i)] = distance

                        fifo_to.append(self.index_right(i))

                        if self.excess[self.index_right(i)] > 0:
                            self.active[self.index_right(i)] = None

            distance = distance + 1

            fifo_temp = fifo_from
            fifo_from = fifo_to
            fifo_to = fifo_temp

    def cut(self, int global_relabel_interval):
        cdef:
            int i

        i = 0
        while len(self.active) > 0:
            self.push()

            if i%global_relabel_interval == 0:
                self.global_relabel()
            else:
                self.relabel()

            i += 1

def graphcut(img, src, sink, global_relabel_interval, weight_lambda=None):
    gc = GraphCut(img, src, sink, weight_lambda)
    gc.cut(global_relabel_interval)

    shape = (img.shape[0], img.shape[1])
    cut = np.zeros(shape)
    height = np.array(gc.height).reshape(shape)

    cut[height == gc.max_label] = 1

    return cut

True