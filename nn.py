import numpy as np
import pandas as pd

def leave_one_out(file, feature_indices=None, k=7):
    if file.endswith('.csv'):
        df = pd.read_csv(file)
        data = df.select_dtypes(include=[np.number]).values
    else:
        data = np.loadtxt(file)

    labels = data[:, 0]
    features = data[:, 1:]

    if feature_indices is None or len(feature_indices) == 0:
        feature_indices = list(range(features.shape[1]))

    sub = features[:, feature_indices]
    correct = 0
    n = len(sub)
    for i in range(n):
        distances = np.sqrt(np.sum((sub - sub[i]) ** 2, axis=1))
        distances[i] = float('inf')
        neighbor_labels = labels[np.argsort(distances)[:k]]
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        if unique[np.argmax(counts)] == labels[i]:
            correct += 1
    return correct / n