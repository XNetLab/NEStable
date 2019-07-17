#-* coding: UTF-8 -*-
#稳定性定义
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import random

def mostSimilar(embeddingSpace,word,topn=10):
    wordRep = embeddingSpace[word].reshape(1,-1)
    words = []
    sims = []
    for key,value in embeddingSpace.items():
        if key == word:
            continue
        words.append(key)
        print(wordRep)
        #sims.append(np.linalg.norm(wordRep-value.reshape(1,-1)))
        sims.append(cosine_similarity(wordRep,value.reshape(1,-1)))
    sortedList = [list(x) for x in zip(*sorted(zip(words, sims), key=lambda pair: pair[1], reverse=False))]
    return sortedList[0][:topn]


def overall_stability(nodes,mostSimilar1,mostSimilar2,topn=10):
    overall = 0.0
    stability = {}
    for node in nodes:
        similar = stability_node(mostSimilar1[node],mostSimilar2[node])
        stability[node] = similar
        overall += similar
    overall = overall/len(nodes)
    return overall,stability



def stability_node(similar1,similar2):
    topn=10
    intersection = set(similar1) & set(similar2)
    a = 1.0 * len(intersection) / topn
    return a
