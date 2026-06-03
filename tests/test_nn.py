from nn import leave_one_out
import numpy as np
import pandas as pd

# Get test files
small_file = "../CS170_Small_DataSet__13.txt"
large_file = "../CS170_Large_DataSet__2.txt"

# Small test
print(f"Small dataset:")
accuracy = leave_one_out(small_file)
print(f"{accuracy * 100:.2f}%\n")

# Large test
print(f"Large dataset:")
accuracy = leave_one_out(large_file)
print(f"{accuracy * 100:.2f}%")
