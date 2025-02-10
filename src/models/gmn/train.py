import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np

# Load the feature and ground truth datasets
feature_matrix_path = "./dataset/PROTEIN/protein_feature_matrix"
ground_truth_path = "./dataset/PROTEIN/protein_ground_truth"

# Load the datasets
feature_matrix = pd.read_csv(feature_matrix_path, header=None)
ground_truth = pd.read_csv(ground_truth_path, header=None)

# Debug: Print the first few rows of the ground truth dataset
print("Ground truth dataset sample:")
print(ground_truth.head())

if ground_truth.shape[1] < 2:
    print("Warning: Ground truth dataset has less than two columns. Assuming self-comparison with a single column of indices.")

# Step 1: Data Preprocessing
def create_graph_data(features):
    """
    Create a graph from the feature matrix row.
    Assumes features are node attributes; edge connections are generated naively.
    """
    # Ensure features are numerical and handle missing values
    features = pd.to_numeric(features, errors='coerce').fillna(0).values
    num_nodes = features.shape[0]
    x = torch.tensor(features, dtype=torch.float).unsqueeze(1)  # Node features

    # Create a fully connected graph
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()

    return Data(x=x, edge_index=edge_index)

# Create graph datasets from feature matrix
graph_list = []
for i in range(feature_matrix.shape[0]):
    graph_data = create_graph_data(feature_matrix.iloc[i])
    graph_list.append(graph_data)

# Step 2: Define the GMN Model
class GraphMatchingNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphMatchingNetwork, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, data1, data2):
        # First graph embedding
        x1 = self.conv1(data1.x, data1.edge_index).relu()
        x1 = self.conv2(x1, data1.edge_index)
        
        # Second graph embedding
        x2 = self.conv1(data2.x, data2.edge_index).relu()
        x2 = self.conv2(x2, data2.edge_index)
        
        # Global pooling (mean pooling)
        h1 = torch.mean(x1, dim=0)
        h2 = torch.mean(x2, dim=0)

        # Compute similarity score
        similarity = self.fc(torch.abs(h1 - h2))
        return similarity

# Step 3: Training Loop
def train_model(graph_list, ground_truth, epochs=20, lr=0.01, model_save_path="gmn_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GraphMatchingNetwork(in_channels=1, hidden_channels=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, row in ground_truth.iterrows():
            try:
                if ground_truth.shape[1] >= 2:
                    idx1, idx2 = int(row.iloc[0]), int(row.iloc[1])
                else:
                    idx1 = int(row.iloc[0])
                    idx2 = idx1  # Assume self-comparison for single-column ground truth
                
                label = torch.tensor([1.0], device=device) if not pd.isna(idx2) else torch.tensor([0.0], device=device)

                if idx1 >= len(graph_list) or idx2 >= len(graph_list):
                    print(f"Skipping invalid graph indices: idx1={idx1}, idx2={idx2}")
                    continue

                graph1 = graph_list[idx1].to(device)
                graph2 = graph_list[idx2].to(device)

                # Forward pass
                output = model(graph1, graph2)
                loss = loss_fn(output, label)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            except Exception as e:
                print(f"Error processing row {i}: {e}")

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")
    return model

# Train the model
trained_model = train_model(graph_list, ground_truth)
