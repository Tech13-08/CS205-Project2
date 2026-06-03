import numpy as np
import pandas as pd

def leave_one_out(file):
    # Handle both csv and txt file types
    if file.endswith('.csv'):
        df = pd.read_csv(file, header=None)
        data = df.select_dtypes(include=[np.number]).values
    else:
        data = np.loadtxt(file)
    
    # Setting main variables for KNN
    k=7
    labels = data[:, 0]
    num_features = data.shape[1] - 1
    features = data[:, 1:][:, list(range(num_features))]
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