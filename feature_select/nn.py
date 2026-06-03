from nn import leave_one_out
import numpy as np
import pandas as pd

def leave_one_out(data, indexes=None):
    # Setting main variables for KNN
    k=7
    labels = data[:, 0]
    if indexes is None:
        features = data[:, 1:]
    else:
        features = data[:, 1:][:, indexes]
    correct = 0

    # KNN Loop for each data point
    for i in range(len(features)):
        distances = np.sqrt(np.sum((features - features[i]) ** 2, axis=1))
        distances[i] = float('inf')
        neighbor_labels = labels[np.argsort(distances)[:k]]
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        if unique[np.argmax(counts)] == labels[i]:
            correct += 1
    
    # Calculate and return accuracy
    return correct / len(features)