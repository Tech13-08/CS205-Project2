import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from nn.load import load_data

def cluster(file_path):
    # Load data
    data = load_data(file_path)
    features = data[:, 1:] 
    drivers = data[:, 0]

    features = features.astype(float)
    
    total_instances = features.shape[0]

    
    # Z-score normalization 
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)

    std[std == 0] = 1.0
    normalized_data = (features - mean) / std
    
    
    leaf_labels = [f"{driver}" for driver in drivers]
    
    print(f"\nDataset loaded with {total_instances} instances.")
    print(f"Constructing Ward's Linkage for {total_instances} elements...")

    # Ward's Linkage
    Z = linkage(normalized_data, method='ward')

    # Render
    plt.figure(figsize=(22, 10))
    dendrogram(
        Z,
        labels=leaf_labels,
        leaf_rotation=90,
        leaf_font_size=12
    )

    plt.title("Dendrogram", fontsize=18)
    plt.xlabel("Driver_TyreCompound", fontsize=14)
    plt.ylabel("Ward Distance", fontsize=14)
    plt.tight_layout()

    output_img = "dendrogram/f1_dendrogram.png"
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    print(f"Saved as {output_img}")
    plt.show()


print("Welcome to the Dendrogram Builder.")
file_choice = input("Type in the name of the file to build for: ").strip()
cluster("dendrogram/" + file_choice)