#-* coding: UTF-8 -*-
#import graph
#import dngr
from dngr.graph import *
from dngr.dngr import DNGR
def train(network_pp, algorithm_pp):
    input = network_pp['name']
    dimensions = algorithm_pp['dimensions']
    kstep = algorithm_pp['kstep']
    layer_num = algorithm_pp['layer_num']
    G = Graph()
    G.read_edgelist(filename=input,weighted=False,directed=False)

    model = DNGR(graph = G, Kstep=kstep, dim=dimensions,layer_num=layer_num)
    return model

def main(input='D:/embeddingStability/data/network/karate.edges',weighted=False,directed=False,dims=64,layer_num=2,kstep = 5):
    G = Graph()
    G.read_edgelist(filename=input, weighted=weighted, directed=directed)
    model = DNGR(graph=G, Kstep=kstep, dim=dims,layer_num=layer_num)
    return model

if __name__ == '__main__':
    model = main()
    dict,nodes = model.save_embeddings()
    print(dict['34'])