import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np

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


def load_model(model_path):
    """
    Load the pre-trained GMN model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphMatchingNetwork(in_channels=1, hidden_channels=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def create_graph_data(features):
    """
    Create a graph from the feature matrix row.
    Assumes features are node attributes; edge connections are generated naively.
    """
    features = pd.to_numeric(features, errors='coerce').fillna(0).values
    num_nodes = features.shape[0]
    x = torch.tensor(features, dtype=torch.float).unsqueeze(1)  # Node features

    # Create a fully connected graph
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def load_protein_data(feature_matrix_path):
    """
    Load the feature matrix and convert each row to a graph.
    """
    feature_matrix = pd.read_csv(feature_matrix_path, header=None)
    graph_list = []
    for i in range(feature_matrix.shape[0]):
        graph_data = create_graph_data(feature_matrix.iloc[i])
        graph_list.append(graph_data)
    return graph_list


def predict_similarity(model, graph_list, idx1, idx2):
    """
    Predict the similarity between two graphs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure indices are valid
    if idx1 >= len(graph_list) or idx2 >= len(graph_list):
        print(f"Invalid indices: idx1={idx1}, idx2={idx2}")
        return None

    graph1 = graph_list[idx1].to(device)
    graph2 = graph_list[idx2].to(device)

    with torch.no_grad():
        similarity_score = model(graph1, graph2).item()

    print(f"Similarity score between graph {idx1} and graph {idx2}: {similarity_score:.4f}")
    return similarity_score


if __name__ == "__main__":
    # Paths and parameters
    feature_matrix_path = "./dataset/PROTEIN/protein_feature_matrix"
    model_path = "gmn_model.pth"

    # Load the pre-trained model
    model = load_model(model_path)

    # Load the PROTEIN graph data
    graph_list = load_protein_data(feature_matrix_path)

    # Example prediction between two graphs (indices 0 and 1)
    idx1, idx2 = 0, 1
    predict_similarity(model, graph_list, idx1, idx2)