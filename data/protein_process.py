import pandas as pd
import os

# Paths to raw data
input_path = "/root/gmb/data/raw/PROTEIN/PROTEINS/PROTEINS_node_attributes.txt"
output_path = "/root/gmb/dataset/PROTEIN/protein_feature_matrix"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load and save the feature matrix
try:
    feature_matrix = pd.read_csv(input_path, header=None, delim_whitespace=True)
    feature_matrix.to_csv(output_path, header=False, index=False)
    print(f"Feature matrix saved to: {output_path}")
except Exception as e:
    print(f"Error processing feature matrix: {e}")

