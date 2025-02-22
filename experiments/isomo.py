import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher


def graph_similarity_score(G1, G2):
    """
    Computes the similarity score between two graphs based on structural isomorphism.

    Args:
        G1, G2: Two NetworkX graphs to compare.

    Returns:
        is_isomorphic (bool): Whether the graphs are isomorphic.
        similarity_score (float): A score between 0 and 1 representing the structural similarity.
    """
    # Check graph isomorphism
    GM = GraphMatcher(G1, G2)
    is_isomorphic = GM.is_isomorphic()

    # Count matching edges and nodes (if isomorphic, all should match)
    matching_edges = len(list(GM.mapping.items()))
    matching_nodes = sum(1 for n1, n2 in GM.mapping.items() if n1 in G1.nodes and n2 in G2.nodes)

    # Calculate the similarity score
    total_edges = len(G1.edges) + len(G2.edges)
    total_nodes = len(G1.nodes) + len(G2.nodes)
    similarity_score = (matching_edges + matching_nodes) / (total_edges + total_nodes)

    return is_isomorphic, similarity_score


if __name__ == "__main__":
    # Create example graphs
    G1 = nx.Graph()
    G1.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

    G2 = nx.Graph()
    G2.add_edges_from([(5, 6), (6, 7), (7, 8), (8, 5)])

    # Compute similarity
    isomorphic, score = graph_similarity_score(G1, G2)
    print(f"Are the graphs isomorphic? {isomorphic}")
    print(f"Similarity score: {score:.2f}")
