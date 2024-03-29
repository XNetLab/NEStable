#-* coding: UTF-8 -*-
from struc2vec.algorithms import *
from struc2vec.algorithms_distances import *

class Graph():
	def __init__(self, g, is_directed, workers, K, S):

		self.G = g.gToDict()
		self.num_vertices = g.number_of_nodes()
		self.num_edges = g.number_of_edges()
		self.is_directed = is_directed
		self.workers = workers
		self.K = K
		self.S = S
	def preprocess_neighbors_with_rw(self):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(exec_rw,self.G,self.workers,self.K,self.S)
			self.degree_list = job.result()

		return

	def create_vectors(self):
		degrees = {}
		degrees_sorted = set()
		G = self.G
		for v in list(G.keys()):
			degree = len(G[v])
			degrees_sorted.add(degree)
			if(degree not in degrees):
				degrees[degree] = {}
				degrees[degree]['vertices'] = deque() 
			degrees[degree]['vertices'].append(v)
		degrees_sorted = np.array(list(degrees_sorted),dtype='int')
		degrees_sorted = np.sort(degrees_sorted)

		l = len(degrees_sorted)
		for index, degree in enumerate(degrees_sorted):
			if(index > 0):
				degrees[degree]['before'] = degrees_sorted[index - 1]
			if(index < (l - 1)):
				degrees[degree]['after'] = degrees_sorted[index + 1]

		self.degrees_vector = degrees

	def calc_distances(self):

		futures = {}
		results = {}

		G = self.G
		number_vertices = len(G)

		vertices_ = list(G.keys())
		vertices_nbrs = get_neighbors(vertices_,G,self.degrees_vector,number_vertices)
		chunks_vertices = partition(vertices_,self.workers)

		distances = {}

		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 0
			for c in chunks_vertices:
				#print c,part
				#print "part",part
				#calc_distances(c, self.degree_list, vertices_nbrs)
				job = executor.submit(calc_distances, c, self.degree_list, vertices_nbrs)
				futures[job] = part
				part += 1

			for job in as_completed(futures):
				part = futures[job]
				distances[part] = job.result()



		self.distances = distances
		self.vertices_nbrs = vertices_nbrs
		return


	def create_distances_network(self):
		chunks_vertices = partition(list(self.G.keys()),self.workers)

		self.multi_graph,self.alias_method_j,self.alias_method_q,self.amount_neighbours = \
		generate_distances_network(self.distances,chunks_vertices,self.vertices_nbrs,self.S)

		return

	def simulate_walks(self,num_walks,walk_length):

		return generate_random_walks(num_walks,walk_length,self.workers,
			list(self.G.keys()),self.multi_graph,
			self.alias_method_j,self.alias_method_q,self.amount_neighbours)