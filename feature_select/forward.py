import numpy as np
import pandas as pd
from .nn import leave_one_out

def forward_selection(data):
    num_features = data.shape[1] - 1
    current = []
    best_overall = 0.0
    best_subset = []
    k = 7
    print("Beginning search")
    for _ in range(num_features):
        best_acc = 0.0
        pick = None
        for f in range(num_features):
            if f in current:
                continue
            subset = current + [f]
            acc = leave_one_out(data, subset)
            print(f"Using feature (s) {subset} -> Accuracy is {acc*100:.2f}%")
            if acc > best_acc:
                best_acc = acc
                pick = f
        if pick is None:
            break
        current.append(pick)
        print(f"Feature set {current} was the best, accuracy is {best_acc*100:.2f}%")
        if best_acc > best_overall:
            best_overall = best_acc
            best_subset = list(current)
        else:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
    print("Finished search!! The best feature subset is", best_subset, "with an accuracy of", best_overall*100, "%")
    return best_subset, best_overall
