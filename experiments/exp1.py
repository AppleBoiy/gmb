from data.utils import load_graph, preprocess_graph
from src.models.baseline import simple_matching
from src.evaluation import matching_accuracy

graph1 = preprocess_graph(load_graph('data/raw/graph1.edgelist'))
graph2 = preprocess_graph(load_graph('data/raw/graph2.edgelist'))

predicted = simple_matching(graph1, graph2)
ground_truth = [0, 1, 2, 3]

accuracy = matching_accuracy(predicted, ground_truth)
print(f'Matching Accuracy: {accuracy:.2f}')