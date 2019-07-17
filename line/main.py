#-* coding: UTF-8 -*-
from line import graph
from line import line

def train(network_pp, algorithm_pp):
    input = network_pp['name']
    dimensions = algorithm_pp['dimensions']
    order = algorithm_pp['order']
    negative_ratio = algorithm_pp['negative_ratio']
    g = graph.Graph()
    g.read_edgelist(filename=input, weighted=False, directed=False)

    model = line.LINE(g, rep_size=dimensions, order=order,negative_ratio=negative_ratio)
    return model


def main(input='../data/karate_34_77.edges',weighted=False, directed=False, dimensions=64,
         order=3,negative_ratio=5):
    g = graph.Graph()
    g.read_edgelist(filename=input, weighted=weighted, directed=directed)
    model = line.LINE(g, rep_size=dimensions, order=order, negative_ratio=negative_ratio)
    return model
