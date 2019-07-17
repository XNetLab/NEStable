#! /usr/bin/env python
# -*- coding: utf-8 -*-
#deepwalk#
import random
from deepwalk import graph
from gensim.models import Word2Vec

def train(network_pp, algorithm_pp,window_size=5,workers=8,iter=1,seed=2018):

  input = network_pp['name']
  dimensions = algorithm_pp['dimensions']

  num_walks = algorithm_pp['walk_number']
  walk_length = algorithm_pp['walk_length']

  G = graph.load_edgelist(input, undirected=False)


  #print("Walking...")
  walks = graph.build_deepwalk_corpus(G, num_paths=num_walks,
                                        path_length=walk_length, alpha=0, rand=random.Random(seed))
  #print("Training...")

  model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, hs=1, workers=workers,iter=iter)
  #model.wv.save_word2vec_format(output)
  return model



def main(input='../data/network/karate.edges',output='embs.txt',undirected=True,dimensions=64,num_walks=10,walk_length=40,
         window_size=5,seed=0,workers=8,iter=1):
  G = graph.load_edgelist(input, undirected=undirected)
  walks = graph.build_deepwalk_corpus(G, num_paths=num_walks,
                                      path_length=walk_length, alpha=0, rand=random.Random(seed))
  model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, hs=1, workers=workers, iter=iter)
  #model.wv.save_word2vec_format(output)
  return model

if __name__ == '__main__':
  main()