#-* coding: UTF-8 -*-
import warnings
from gensim.models import Word2Vec
from struc2vec import graph
from struc2vec import struc

def train(network_pp, algorithm_pp,window_size=5,workers=8,iter=1):
	input = network_pp['name']
	undirected = True

	dimensions = algorithm_pp['dimensions']
	num_walks = algorithm_pp['walk_number']
	walk_length = algorithm_pp['walk_length']
	S = algorithm_pp['k']
	K = 10
	G = graph.load_edgelist(input, undirected=undirected)
	if undirected:
		directed = False
	else:
		directed = True
	G = struc.Graph(G, directed, workers, K, S)
	G.preprocess_neighbors_with_rw()
	G.create_vectors()
	G.calc_distances()
	G.create_distances_network()
	walks = G.simulate_walks(num_walks, walk_length)

	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, hs=1, sg=1, workers=workers, iter=iter)
	return model


def main(input='D:/embeddingStability/data/network/karate.edges',dimensions=64,undirected=True,num_walks=10,walk_length=40,
				   window_size=5,workers=8,iter=5,K = 10,S=13):
	G = graph.load_edgelist(input, undirected=undirected)
	if undirected:
		directed = False
	else:
		directed = True
	G = struc.Graph(G, directed,workers, K, S)
	G.preprocess_neighbors_with_rw()
	G.create_vectors()
	G.calc_distances()
	G.create_distances_network()
	walks = G.simulate_walks(num_walks, walk_length)

	walks = [list(map(str,walk)) for walk in walks]
	model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, hs=1, sg=1, workers=workers, iter=iter)
	return model
	#model.wv.save_word2vec_format(output)

if __name__ == "__main__":
	main()

