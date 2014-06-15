import netwrokx as nx

class Graph(nx.Graph):

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
