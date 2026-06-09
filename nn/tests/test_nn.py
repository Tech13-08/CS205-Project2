from ..nn import leave_one_out
from ..load import load_data

# Get test files (relative)
small_file = "CS170_Small_DataSet__13.txt"
large_file = "CS170_Large_DataSet__2.txt"

small_data = load_data(small_file)
large_data = load_data(large_file)

print('Small dataset:')
accuracy = leave_one_out(small_data)
print(f"{accuracy * 100:.2f}%\n")

print('Large dataset:')
accuracy = leave_one_out(large_data)
print(f"{accuracy * 100:.2f}%")
