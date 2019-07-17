#-* coding: UTF-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from time import time
from collections import deque
import numpy as np
import math,random
from concurrent.futures import ProcessPoolExecutor, as_completed

def chooseNeighbor(v,graphs,alias_method_j,alias_method_q,layer):
    v_list = graphs[v][-1].astype(int)

    idx = alias_draw(alias_method_j[v][layer].astype(int),alias_method_q[v][layer])
    v = v_list[idx]

    return v


def exec_random_walk(multi_graph,alias_method_j,alias_method_q,v,walk_length,amount_neighbours):
    root = v
    t0 = time()
    layer = 0

    path = []
    path.append(v)

    while len(path) < walk_length:
        r = random.random()

        if(r < 0.3):
                v = chooseNeighbor(v,multi_graph,alias_method_j,alias_method_q,layer)
                path.append(v)

        else:
            r = random.random()
            limiar_moveup = prob_moveup(amount_neighbours[layer][v])
            if(r > limiar_moveup):
                if(layer > 0):
                    layer = layer - 1           
            else:
                if((layer + 1) < len(multi_graph[v]) - 1):
                    layer = layer + 1
    t1 = time()
    return path


def exec_ramdom_walks_for_chunck(vertices,multi_graph,alias_method_j,alias_method_q,walk_length,amount_neighbours):
    walks = deque()
    for v in vertices:
        walks.append(exec_random_walk(multi_graph,alias_method_j,alias_method_q,v,walk_length,amount_neighbours))
    return walks

def generate_random_walks(num_walks,walk_length,workers,vertices,multi_graph,alias_method_j,alias_method_q,amount_neighbours):

    t0 = time()
    
    walks = []

    if(workers > num_walks):
        workers = num_walks

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for walk_iter in range(num_walks):
            random.shuffle(vertices)
            job = executor.submit(exec_ramdom_walks_for_chunck,vertices,multi_graph,alias_method_j,alias_method_q,walk_length,amount_neighbours)
            futures[job] = walk_iter
        for job in as_completed(futures):
            walk = job.result()
            r = futures[job]
            walks.extend(walk)
            del futures[job]


    t1 = time()
    return walks



def prob_moveup(amount_neighbours):
    x = math.log(amount_neighbours + math.e)
    p = (x / ( x + 1))
    return p


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
