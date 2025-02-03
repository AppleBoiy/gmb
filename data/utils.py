import networkx as nx
import numpy as np

def load_graph(file_path):
    return nx.read_edgelist(file_path, nodetype=int)

def preprocess_graph(graph):
    for u, v, data in graph.edges(data=True):
        data['weight'] = data.get('weight', 1.0)
    return graph