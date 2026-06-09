import pandas as pd
from .forward import forward_selection
from .backward import backward_elimination
from nn.nn import leave_one_out
from nn.load import load_data

print("Welcome to the Feature Selection Algorithm.")
file_choice = input("Type in the name of the file to test: ").strip()
data = load_data("feature_select/" + file_choice)
baseline = leave_one_out(data)

print("Type in the number of the algorithm you want to run.")
print("1) Forward Selection")
print("2) Backward Elimination")
choice = input("").strip()

print(f"This dataset has {data.shape[1] - 1} features (not including the class attribute), with {data.shape[0]} instances.")
print(f"Running nearest neighbor with all {data.shape[1] - 1} features, using \"Leave-One-Out\" evaluation, accuracy is {baseline*100:.2f}%")

if choice == "1":
    forward_selection(data)
else:
    backward_elimination(data)
