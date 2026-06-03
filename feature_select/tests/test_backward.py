from ..backward import backward_elimination
from ..load import load_data

small = "feature_select/CS170_Small_DataSet__13.txt"
large = "feature_select/CS170_Large_DataSet__2.txt"

small_data = load_data(small)
large_data = load_data(large)

print('SMALL DATASET')
backward_elimination(small_data)

print('\nLARGE DATASET')
backward_elimination(large_data)
