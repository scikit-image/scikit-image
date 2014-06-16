import networkx as nx
import _construct
from skimage import util

class RAG(nx.Graph):

    def merge_nodes(i,j):
        if not self.has_edge(i, j):
            raise ValueError('Cant merge non adjacent nodes')

        # print "before ",self.order()
        for x in self.neighbors(i):
            if x == j:
                continue
            w1 = self.get_edge_data(x, i)['weight']
            w2 = -1
            if self.has_edge(x, j):
                w2 = self.get_edge_data(x, j)['weight']

            w = max(w1, w2)

            self.add_edge(x, j, weight=w)

        self.node[j]['labels'] += self.node[i]['labels'] 
        self.remove_node(i)

def rag_meancolor(img,labels):

    img = util.img_as_ubyte(img)
    if img.ndim == 4 :
        return _construct.construct_rag_meancolor_3d(img,labels)
    elif img.ndim == 3 :
        return _construct.construct_rag_meancolor_2d(img,labels)
    else :
        raise ValueError("Image dimension not supported")
