import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics.pairwise import cosine_similarity


# Step 1: Define Graph Data
def create_graph(node_features, edge_index, label=None):
    return Data(x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t(),
                y=torch.tensor([label] if label else [], dtype=torch.long))


# Graph 1
graph1 = create_graph(
    node_features=[
        [1, 0],  # Node 0 feature
        [0, 1],  # Node 1 feature
        [1, 1],  # Node 2 feature
    ],
    edge_index=[
        [0, 1], [1, 0],
        [1, 2], [2, 1],
    ],
)

# Graph 2
graph2 = create_graph(
    node_features=[
        [0, 1],  # Node 0 feature
        [1, 1],  # Node 1 feature
        [1, 0],  # Node 2 feature
    ],
    edge_index=[
        [0, 1], [1, 0],
        [1, 2], [2, 1],
    ],
)


# Step 2: Define the Graph Neural Network
class GraphSimNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSimNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)


# Step 3: Generate Graph Embeddings
def get_graph_embedding(graph_data, model):
    # Add batch information for processing a single graph
    graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
    with torch.no_grad():
        return model(graph_data.x, graph_data.edge_index, graph_data.batch)


# Step 4: Predict Graph Similarity
def predict(graph1, graph2):
    """
    Predict the similarity between two graphs using a Graph Neural Network (GNN) and cosine similarity.

    Args:
        graph1: First graph (PyG Data object).
        graph2: Second graph (PyG Data object).

    Returns:
        similarity_score: Cosine similarity score between the embeddings of the two graphs.
    """
    # Initialize the model
    input_dim = graph1.x.size(1)  # Number of features per node
    hidden_dim = 16
    output_dim = 8

    model = GraphSimNet(input_dim, hidden_dim, output_dim)
    model.eval()  # Set model to evaluation mode

    # Generate embeddings for both graphs
    embedding1 = get_graph_embedding(graph1, model)
    embedding2 = get_graph_embedding(graph2, model)

    # Compute cosine similarity
    similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    return similarity[0][0]


# Example Usage
if __name__ == "__main__":
    # Predict similarity
    similarity_score = predict(graph1, graph2)
    print(f"Predicted Graph Similarity Score: {similarity_score:.4f}")

