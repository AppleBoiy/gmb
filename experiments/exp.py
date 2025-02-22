import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from networkx.algorithms.similarity import graph_edit_distance
import time
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


# Graph Isomorphism
def graph_similarity_isomorphism(G1, G2):
    start_time = time.time()
    GM = GraphMatcher(G1, G2)
    is_isomorphic = GM.is_isomorphic()
    time_used = time.time() - start_time
    return is_isomorphic, time_used


# Graph Edit Distance
def graph_similarity_ged(G1, G2):
    start_time = time.time()
    ged = graph_edit_distance(G1, G2)
    max_possible_edits = len(G1.nodes) + len(G2.nodes) + len(G1.edges) + len(G2.edges)
    similarity_score = 1 - (ged / max_possible_edits)
    time_used = time.time() - start_time
    return similarity_score, time_used


# Node/Edge-based Similarity
def node_edge_similarity(G1, G2):
    start_time = time.time()
    common_nodes = len(set(G1.nodes) & set(G2.nodes))
    common_edges = len(set(G1.edges) & set(G2.edges))

    similarity_score = (common_nodes + common_edges) / (len(G1.nodes) + len(G1.edges) + len(G2.nodes) + len(G2.edges))
    time_used = time.time() - start_time
    return similarity_score, time_used


# Convert NetworkX graph to PyTorch Geometric Data format
def nx_to_data(G):
    time_start = time.time()
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.ones(G.number_of_nodes(), 1)  # Node features (1-dimensional)
    time_used = time.time() - time_start
    return Data(x=x, edge_index=edge_index), time_used


# Graph Kernel (GCN-based)
def graph_similarity_gcn(G1, G2):
    start_time = time.time()
    data1, _ = nx_to_data(G1)  # Unpack Data object and time used
    data2, _ = nx_to_data(G2)  # Unpack Data object and time used

    class GCN(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(in_channels, out_channels)
            self.conv2 = GCNConv(out_channels, out_channels)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x

    # Initialize GCN model
    model = GCN(1, 2)
    out1 = model(data1)
    out2 = model(data2)

    # Compute similarity (e.g., using cosine similarity)
    similarity_score = torch.cosine_similarity(out1.mean(dim=0), out2.mean(dim=0), dim=0)
    time_used = time.time() - start_time
    return similarity_score.item(), time_used


# Benchmark the similarity methods
def benchmark_graph_similarity(G1, G2):
    # Graph Isomorphism
    is_isomorphic, iso_time = graph_similarity_isomorphism(G1, G2)
    print(f"Isomorphic: {is_isomorphic}, Time: {iso_time:.4f}s")

    # Graph Edit Distance
    ged_similarity, ged_time = graph_similarity_ged(G1, G2)
    print(f"Graph Edit Distance Similarity: {ged_similarity:.4f}, Time: {ged_time:.4f}s")

    # Node/Edge-based Similarity
    node_edge_sim, ned_time = node_edge_similarity(G1, G2)
    print(f"Node/Edge Similarity: {node_edge_sim:.4f}, Time: {ned_time:.4f}s")

    # GCN-based Graph Kernel Similarity
    gcn_similarity, gcn_time = graph_similarity_gcn(G1, G2)
    print(f"GCN-based Graph Kernel Similarity: {gcn_similarity:.4f}, Time: {gcn_time:.4f}s")


# Example usage
if __name__ == "__main__":
    # Create sample graphs
    G1 = nx.erdos_renyi_graph(10, 0.5)
    G2 = nx.erdos_renyi_graph(10, 0.5)

    benchmark_graph_similarity(G1, G2)
