#-* coding: UTF-8 -*-
#import graph
#import sdne
from sdne.graph import *
from sdne.sdne import SDNE
def train(network_pp, algorithm_pp):
    input = network_pp['name']
    dimensions = algorithm_pp['dimensions']
    layer_num = algorithm_pp['layer_num']
    neural_num = algorithm_pp['neural_num']
    G = Graph()
    G.read_edgelist(filename=input,weighted=False,directed=False)
    encoder_layer_list = []
    for i in range(layer_num - 1):
        encoder_layer_list.append(neural_num)
    encoder_layer_list.append(dimensions)

    model = SDNE(G, encoder_layer_list=encoder_layer_list)
    return model

def main(input='D:/embeddingStability/data/network/karate.edges',weighted=False,directed=False,dims=64,layer_num=2,neural_num=100):
    G = Graph()
    G.read_edgelist(filename=input, weighted=weighted, directed=directed)
    encoder_layer_list = []
    for i in range(layer_num-1):
        encoder_layer_list.append(neural_num)
    encoder_layer_list.append(dims)
    model = SDNE(G, encoder_layer_list=encoder_layer_list)
    return model


if __name__ == '__main__':
    model = main()
    dict,nodes = model.save_embeddings()
    print(dict['26'])