import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def simple_matching(graph1, graph2):
    adj1 = nx.adjacency_matrix(graph1).toarray()
    adj2 = nx.adjacency_matrix(graph2).toarray()
    similarity = cosine_similarity(adj1, adj2)
    return np.argmax(similarity, axis=1)
