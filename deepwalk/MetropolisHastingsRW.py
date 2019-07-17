#-* coding: UTF-8 -*-
import random
import networkx as nx
from gensim.models import Word2Vec
class Graph():
    def __init__(self,nx_G):
        self.G = nx_G
        self.nodeNeighbor()

    def nodeNeighbor(self):
        self.node_neighbor = {}
        for e in  self.G.edges():
            if e[0] not in self.node_neighbor: self.node_neighbor[e[0]] = set()
            if e[1] not in self.node_neighbor: self.node_neighbor[e[1]] = set()
            self.node_neighbor[e[0]].add(e[1])
            self.node_neighbor[e[1]].add(e[0])

    def build_deepwalk_corpus(self, num_paths = 10, path_length = 40,random_method = 'MHRW'):
        walks = []
        nodes = list(self.G.nodes())
        if random_method == 'MHRW':
            self.method = self.MetropolisHastingsRW
        elif random_method == 'RWRW':
            self.method = self.ReWeightedRW
        for cnt in range(num_paths):
            random.shuffle(nodes)
            # rand.shuffle(nodes)
            for node in nodes:
                walks.append(self.method(path_length,start=node))
        return walks

    def ReWeightedRW(self,path_length,start=None):
        sampling = list()
        node = start
        node_degrees = list()
        while len(sampling) < path_length:
            sampling.append(node)
            node_degrees.append(len(self.node_neighbor[node]))
            node = random.sample(self.node_neighbor.get(node), 1)[0]

        normalization_constant = 0.0
        for x in node_degrees:
            normalization_constant += (1.0 / x)

        prob = list()
        for x in node_degrees:
            temp = (1.0 / x) / normalization_constant
            prob.append(temp)
        print(prob)
        return sampling

    def MetropolisHastingsRW(self, path_length,start=None):
        sampling = list()
        node = start
        sampling.append(node)
        while len(sampling) < path_length:
            neighbor = random.sample(self.node_neighbor.get(node), 1)[0]
            if (len(self.node_neighbor[node]) > len(self.node_neighbor[neighbor])):
                node = neighbor
            else:
                rand = random.random()
                prob = (1.0 * len(self.node_neighbor[node])) / len(self.node_neighbor[neighbor])
                if (rand < prob):
                    node = neighbor
                else:
                    continue
            sampling.append(node)
        return sampling

if __name__ == '__main__':
    nxG = nx.read_edgelist('D:/embeddingStability/data/others/karate.edges')
    G = Graph(nxG)
    walks = G.build_deepwalk_corpus(random_method='RWRW')
    print(len(walks[0]))
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=2, window=5, min_count=0, sg=1, hs=1, workers=3, iter=1)
