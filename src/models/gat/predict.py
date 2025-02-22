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

def predict(graph1, graph2):
    """
    Predict the similarity between two graphs using a pre-trained model.

    Args:
        graph1: First graph (PyG Data object).
        graph2: Second graph (PyG Data object).

    Returns:
        similarity_score: A scalar value between -1 and 1 indicating the similarity between the two graphs.
    """
    # Load the dataset (if node features or structure are dataset-dependent)
    dataset = load_dataset()

    # Load the pre-trained model
    model = GraphMatchingModel(num_node_features=dataset.num_node_features, hidden_dim=64)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()  # Set model to evaluation mode

    # Compute the similarity between the graphs
    similarity_score = predict_similarity(model, graph1, graph2)
    return similarity_score


# Example usage
if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset()

    # Select two graphs for comparison
    graph1 = dataset[0]
    graph2 = dataset[1]  # Use a different graph for comparison

    # Predict similarity
    similarity = predict(graph1, graph2)
    print(f"Predicted similarity between graph1 and graph2: {similarity:.4f}")
