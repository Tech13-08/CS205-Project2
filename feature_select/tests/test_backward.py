from ..backward import backward_elimination
from nn.load import load_data

small = "CS170_Small_DataSet__13.txt"
large = "CS170_Large_DataSet__2.txt"

small_data = load_data(small)
large_data = load_data(large)

print('SMALL DATASET')
backward_elimination(small_data)

print('\nLARGE DATASET')
backward_elimination(large_data)
