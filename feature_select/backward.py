import numpy as np
from .nn import leave_one_out

def backward_elimination(data):
    num_features = data.shape[1] - 1
    current = list(range(num_features))
    best_overall = 0.0
    best_subset = list(current)
    print("Beginning search")
    for _ in range(num_features - 1):
        best_acc = 0.0
        remove = None
        for f in current:
            subset = [i for i in current if i != f]
            acc = leave_one_out(data, subset)
            print(f"Using feature(s) {subset} -> Accuracy is {acc*100:.2f}%")
            if acc > best_acc:
                best_acc = acc
                remove = f
        if remove is None:
            break
        current.remove(remove)
        print(f"Feature set {current} was the best, accuracy is {best_acc*100:.2f}%")
        if best_acc > best_overall:
            best_overall = best_acc
            best_subset = list(current)
        else:
            print("(Warning, Accuracy decreased after removing feature! Continuing search in case of local maxima)")
    print("Finished search!! The best feature subset is", best_subset, "with an accuracy of", best_overall*100, "%")
    return best_subset, best_overall
