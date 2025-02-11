import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np  # Add this import at the top of the file
import random

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

def create_pairs(dataset, num_pairs=1000):
    pairs = []
    labels = []
    classes = [data.y.item() for data in dataset]
    
    for _ in range(num_pairs):
        idx1, idx2 = random.sample(range(len(dataset)), 2)
        label = 1 if classes[idx1] == classes[idx2] else 0
        pairs.append((dataset[idx1], dataset[idx2]))
        labels.append(label)
    
    return pairs, labels

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

# pairing data
pairs, labels = create_pairs(dataset)

# Initialize model
model = GraphMatchingModel(num_node_features=dataset.num_node_features, hidden_dim=64)
    

# Split pairs into train and test sets
train_pairs, test_pairs, train_labels, test_labels = train_test_split(pairs, labels, test_size=0.2, random_state=42)

# Create DataLoader
train_loader = DataLoader(list(zip(train_pairs, train_labels)), batch_size=32, shuffle=True)
test_loader = DataLoader(list(zip(test_pairs, test_labels)), batch_size=32, shuffle=False)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

# Training loop
def train():
    model.train()
    total_loss = 0
    for (graph1, graph2), labels in train_loader:
        optimizer.zero_grad()
        
        # Get embeddings for both graphs
        emb1 = model(graph1.x, graph1.edge_index, graph1.batch)
        emb2 = model(graph2.x, graph2.edge_index, graph2.batch)
        
        # Compute similarity (dot product)
        similarity = torch.sum(emb1 * emb2, dim=1)
        
        # Compute loss
        loss = criterion(similarity, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Evaluation loop
def test(loader):
    model.eval()
    correct = 0
    for (graph1, graph2), labels in loader:
        emb1 = model(graph1.x, graph1.edge_index, graph1.batch)
        emb2 = model(graph2.x, graph2.edge_index, graph2.batch)
        similarity = torch.sum(emb1 * emb2, dim=1)
        preds = (similarity > 0).float()
        correct += (preds == labels).sum().item()
    
    return correct / len(loader.dataset)

# Train and evaluate
for epoch in range(1, 151):
    loss = train()
    acc = test(test_loader)
    print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'graph_matching_model.pth')
