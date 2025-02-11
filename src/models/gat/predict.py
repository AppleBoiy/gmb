import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset


def predict_similarity(model, graph1, graph2):
    """
    Predict the similarity between two graphs using cosine similarity.
    Args:
        model: Trained GraphMatchingModel.
        graph1: First graph (PyG Data object).
        graph2: Second graph (PyG Data object).
    Returns:
        similarity_score: A scalar value between -1 and 1 indicating the similarity between the two graphs.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Get embeddings for both graphs
    with torch.no_grad():  # Disable gradient computation
        emb1 = model(graph1.x, graph1.edge_index, torch.zeros(graph1.x.size(0), dtype=torch.long))
        emb2 = model(graph2.x, graph2.edge_index, torch.zeros(graph2.x.size(0), dtype=torch.long))
    
    # Ensure embeddings are 1D tensors
    emb1 = emb1.squeeze()  # Remove batch dimension if present
    emb2 = emb2.squeeze()  # Remove batch dimension if present
    
    # Compute cosine similarity
    similarity_score = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1).item()
    return similarity_score

def load_dataset():
    # Load MUTAG dataset
    dataset = TUDataset(root='data/raw/MUTAG', name='MUTAG')

    # Print dataset information
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Number of node features: {dataset.num_node_features}')

    # Example: Inspect the first graph
    data = dataset[0]
    print(data)  # Prints the graph's node features, edge indices, and label

    return dataset

class GraphMatchingModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim):
        super(GraphMatchingModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
    
    def forward(self, x, edge_index, batch):
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Graph-level embedding
        x = global_mean_pool(x, batch)
        return x

# load dataset
dataset = load_dataset()

# Load the model architecture
model = GraphMatchingModel(num_node_features=dataset.num_node_features, hidden_dim=64)

# Load the trained weights
model.load_state_dict(torch.load('graph_matching_model.pth'))

# Set the model to evaluation mode
model.eval()

# Select two graphs from the dataset
graph1 = dataset[0]  # First graph
graph2 = dataset[0]  # Second graph

# Predict similarity
similarity_score = predict_similarity(model, graph1, graph2)
print(graph1, graph2)
print(f'Similarity between graph1 and graph2: {similarity_score:.4f}')