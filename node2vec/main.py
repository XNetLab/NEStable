#-* coding: UTF-8 -*-
import networkx as nx
from node2vec import graph
from gensim.models import Word2Vec

def train(network_pp, algorithm_pp,window_size=5,iter=1,workers=8):

	input = network_pp['name']

	p = algorithm_pp['p']
	q = algorithm_pp['q']
	num_walks = algorithm_pp['walk_number']
	walk_length = algorithm_pp['walk_length']
	dimensions = algorithm_pp['dimensions']
	nx_G = nx.read_edgelist(input)
	for edge in nx_G.edges():
		nx_G[edge[0]][edge[1]]['weight'] = 1
	G = graph.Graph(nx_G, False, p, q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(num_walks,walk_length)
	walks = [list(map(str, walk)) for walk in walks]

	model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, hs=1,workers=workers, iter=iter)
	#model.wv.save_word2vec_format(output)
	return model

def main(input = '../data/karate_34_77.edges',p = 1.0,q = 1.0,weight = False,directed = False,
		 num_walks = 10,walk_length = 40,dimensions = 64,
	window_size = 5,workers =8,iter =1):
	if weight:
		nx_G = nx.read_weighted_edgelist(input, nodetype=int, create_using=nx.DiGraph())
	else:
		nx_G = nx.read_edgelist(input)
		for edge in nx_G.edges():
			nx_G[edge[0]][edge[1]]['weight'] = 1
	if not directed:
		nx_G = nx_G.to_undirected()
	G = graph.Graph(nx_G, directed, p, q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(num_walks, walk_length)
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, hs=1, workers=workers, iter=iter)
	return model
	# model.wv.save_word2vec_format(output)
