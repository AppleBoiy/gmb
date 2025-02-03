def matching_accuracy(predicted, ground_truth):
    correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
    return correct / len(ground_truth)