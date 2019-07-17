#-* coding: UTF-8 -*-
import pandas as pd
import tensorflow as tf
import numpy as np
from lap.lap import LaplacianEigenmaps
from gf.gf import GraphFactorization
from tensorflow.contrib.tensorboard.plugins import projector
import os
import deepwalk.main as deepwalk
import gf.gf as gf
import lap.lap as lap
import dngr.main as dngr
import line.main as line
import node2vec.main as node2vec
import struc2vec.main as struc2vec
import  sdne.main as sdne
from sklearn.metrics.pairwise import cosine_similarity
from stability import  overall_stability
from stability import mostSimilar
import networkx as nx
import random
import numpy as np
import matplotlib as mpl
from lap.lap import LaplacianEigenmaps
from lap.graph import Graph
mpl.use('TkAgg')
import matplotlib.pyplot as plt
#mpl.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False

def train_deepwalk(file):
    model = deepwalk.main(file)
    nodes = [node for node, vcab in model.wv.vocab.items()]
    dict = {}
    for node in nodes:
        dict[node] = model.wv[node]
    print('finished deepwalk')
    return dict
def train_dngr(file):
    model = dngr.main(file)
    dict, nodes = model.save_embeddings()
    print('finished dngr')
    return dict
def train_line(file):
    model = line.main(file)
    dict, nodes = model.save_embeddings()
    print('finished line')
    return dict
def train_node2vec(file):
    model = node2vec.main(file)
    nodes = [node for node, vcab in model.wv.vocab.items()]
    dict = {}
    for node in nodes:
        dict[node] = model.wv[node]
    print('finished node2vec')
    return dict
def train_sdne(file):
    model = sdne.main(file)
    dict, nodes = model.save_embeddings()
    print('finished sdne')
    return dict
def train_struc2vec(file):
    model = struc2vec.main(file)
    nodes = [node for node, vcab in model.wv.vocab.items()]
    dict = {}
    for node in nodes:
        dict[node] = model.wv[node]
    print('finished struc2vec')
    return dict

def train_gf(file):
    g=Graph()
    #g.read_edgelist(file)
    g.read_edgelist(filename= file, weighted=True,directed=True)
    model = GraphFactorization(g)
    vectors = model.vectors
    print('finished gf')
    return vectors

def train_lap(file):
    g=Graph()
    #g.read_edgelist(file)
    g.read_edgelist(filename= file, weighted=True,directed=True)
    model =LaplacianEigenmaps(g)
    vectors = model.vectors

    print('finished lap')
    return vectors

def dict_add(dict1,dict2):
    for k,v in dict1.items():
        if k in dict2.keys():
            dict2[k] = dict2[k]+v
        else:
            dict2[k] = v
    return dict2
def dict_div(dict,num):
    for k,v in dict.items():
        dict[k] = dict[k]/(1.0*num)
    return dict

def test(T,file,nodes):

    model_set = []
    for i in range(T):
        dict = train_deepwalk(file)
        #dict = train_node2vec(file)
        #dict = train_line(file)
        #dict = train_dngr(file)
        #dict = train_sdne(file)
        #dict=train_struc2vec(file)
        #dict = train_gf(file)

        model_set.append(dict)
        # 稳定的grarep,hope


    st = {}
    mostSimilar1={}
    mostSimilar2={}
    mostSimilar3 = {}
    mostSimilar4 = {}
    mostSimilar5 = {}
    for node in nodes:
        mostSimilar1[node]=mostSimilar(model_set[0],node,topn=10)
    for node in nodes:
        mostSimilar2[node]=mostSimilar(model_set[1],node,topn=10)
    for node in nodes:
        mostSimilar3[node]=mostSimilar(model_set[2],node,topn=10)
    for node in nodes:
        mostSimilar4[node]=mostSimilar(model_set[3],node,topn=10)
    for node in nodes:
        mostSimilar5[node]=mostSimilar(model_set[4],node,topn=10)
    all_mostSimilar=[mostSimilar1,mostSimilar2,mostSimilar3,mostSimilar4,mostSimilar5]

    for i in range(T):
        for j in range(i + 1, T):
            over_all, stability = overall_stability(nodes, all_mostSimilar[i], all_mostSimilar[j],topn=10)
            st = dict_add(st, stability)
    st = dict_div(st,T*(T-1)/2.0)
    over_all = 0.0
    print("T = "+str(T))
    print(st)
    for node in nodes:
        over_all += st[node]
    over_all/=len(nodes)
    #st返回每个节点的稳定度，over_all返回所有节点的平均稳定度
    return over_all,st

if __name__ == '__main__':
    # T是向量空间个数
    T=5
    #数据集路径
    file='data/BlogCatalog.txt'
    G=nx.read_edgelist(file)
    #nodes为该数据集中的所有节点
    nodes=G.nodes()
    test(T,file,nodes)
