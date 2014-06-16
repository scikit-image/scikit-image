import rag
cimport numpy as cnp
import numpy as np


def construct_rag_meancolor_3d( img, arr):
    cdef Py_ssize_t l, b, h, i, j, k
    cdef cnp.int32_t current, next
    l = arr.shape[0]
    b = arr.shape[1]
    h = arr.shape[2]

    g = rag.RAG()

    i = 0
    while i < l - 1:
        j = 0
        while j < b - 1:
            k = 0
            while k < h - 1:
                current = arr[i, j, k]
        
                try :
                    g.node[current]['pixel_count'] += 1
                    g.node[current]['total_color'] += img[i,j]
                except KeyError:
                    g.add_node(current)
                    g.node[current]['pixel_count'] = 1
                    g.node[current]['total_color'] = img[i,j].astype(np.long)
                    g.node[current]['labels'] = [arr[i,j]]

                next = arr[i + 1, j, k]
                if current != next:
                    g.add_edge(current, next)

                next = arr[i, j + 1, k]
                if current != next:
                    g.add_edge(current, next)

                next = arr[i + 1, j + 1, k]
                if current != next:
                    g.add_edge(current, next)

                next = arr[i + 1, j, k + 1]
                if current != next:
                    g.add_edge(current, next)

                next = arr[i, j + 1, k + 1]
                if current != next:
                    g.add_edge(current, next)

                next = arr[i + 1, j + 1, k + 1]
                if current != next:
                    g.add_edge(current, next)

                next = arr[i, j, k + 1]
                if current != next:
                    g.add_edge(current, next)


                k += 1

            j += 1

        i += 1


    for n in g.nodes():
        g.node[n]['mean_color'] = g.node[n]['total_color']/g.node[n]['pixel_count']

    for x,y in g.edges_iter() :
        diff = g.node[x]['mean_color'] - g.node[y]['mean_color']
        g[x][y]['weight'] = np.sqrt(diff.dot(diff))

    return g


def construct_rag_meancolor_2d(img, arr):
    cdef Py_ssize_t l, b, h, i, j, k
    cdef cnp.int32_t current, next
    l = arr.shape[0]
    b = arr.shape[1]

    g = rag.RAG()

    i = 0
    while i < l - 1:
        j = 0
        while j < b - 1:
            current = arr[i, j]

            try :
                g.node[current]['pixel_count'] += 1
                g.node[current]['total_color'] += img[i,j]
            except KeyError:
                g.add_node(current)
                g.node[current]['pixel_count'] = 1
                g.node[current]['total_color'] = img[i,j].astype(np.long)
                g.node[current]['labels'] = [arr[i,j]]

            next = arr[i + 1, j]
            if current != next:
                g.add_edge(current, next)

            next = arr[i, j + 1]
            if current != next:
                g.add_edge(current, next)

            next = arr[i + 1, j + 1]
            if current != next:
                g.add_edge(current, next)

            j += 1

        i += 1


    for n in g.nodes():
        g.node[n]['mean_color'] = g.node[n]['total_color']/g.node[n]['pixel_count']

    for x,y in g.edges_iter() :
        diff = g.node[x]['mean_color'] - g.node[y]['mean_color']
        g[x][y]['weight'] = np.sqrt(diff.dot(diff))


    return g
