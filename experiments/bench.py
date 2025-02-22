from exp import *
from src.models.gmn.predict import predict as gmn_predict
from src.models.gat.predict import predict as gat_predict
from src.models.sgc.predict import predict as gat_predict
import time
import networkx as nx

# Convert networkx graph to torch_geometric Data object
def nx_to_data(G):
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.ones(G.number_of_nodes(), 1)  # Node features (1-dimensional)
    return Data(x=x, edge_index=edge_index)


# Benchmark GMN
def benchmark_gmn(G1, G2):
    # Convert networkx graphs to torch_geometric Data objects
    data1 = nx_to_data(G1)
    data2 = nx_to_data(G2)

    start = time.time()
    score = gmn_predict(data1, data2)  # Pass Data objects to the GMN model
    end = time.time()
    return score, end - start

# Benchmark SGC
def benchmark_sgc(G1, G2):
    data1 = nx_to_data(G1)
    data2 = nx_to_data(G2)

    start = time.time()
    score = gat_predict(data1, data2)
    end = time.time()
    return score, end - start


# Benchmark GAT
def benchmark_gat(G1, G2):
    # Convert networkx graphs to torch_geometric Data objects
    data1 = nx_to_data(G1)
    data2 = nx_to_data(G2)

    start = time.time()
    score = gat_predict(data1, data2)  # Pass Data objects to the GAT model
    end = time.time()
    return score, end - start


# Benchmark the similarity methods
def benchmark_graph_similarity(G1, G2):
    # Graph Isomorphism
    is_isomorphic, iso_time = graph_similarity_isomorphism(G1, G2)
    print(f"Isomorphic: {is_isomorphic}, Time: {iso_time:.4f}s")

    # Graph Edit Distance
    ged_similarity, ged_time = graph_similarity_ged(G1, G2)
    print(f"Graph Edit Distance Similarity: {ged_similarity}, Time: {ged_time:.4f}s")

    # Node/Edge-based Similarity
    node_edge_sim, ned_time = node_edge_similarity(G1, G2)
    print(f"Node/Edge Similarity: {node_edge_sim}, Time: {ned_time:.4f}s")

    # GCN-based Graph Kernel Similarity
    gcn_similarity, gcn_time = graph_similarity_gcn(G1, G2)
    print(f"GCN-based Graph Kernel Similarity: {gcn_similarity}, Time: {gcn_time:.4f}s")

    # Benchmark GMN
    gmn_score, gmn_time = benchmark_gmn(G1, G2)
    print(f"GMN Similarity: {gmn_score}, Time: {gmn_time:.4f}s")

    # SGC Similarity
    sgc_score, sgc_time = benchmark_sgc(G1, G2)
    print(f"SGC Similarity: {sgc_score}, Time: {sgc_time:.4f}s")

    # Benchmark GAT
    gat_score, gat_time = benchmark_gat(G1, G2)
    print(f"GAT Similarity: {gat_score}, Time: {gat_time:.4f}s")


if __name__ == '__main__':
    # Create sample graphs
    G1 = nx.erdos_renyi_graph(10, 0.5)
    G2 = nx.erdos_renyi_graph(10, 0.5)

    benchmark_graph_similarity(G1, G2)
