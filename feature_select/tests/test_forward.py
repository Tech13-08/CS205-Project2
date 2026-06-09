from ..forward import forward_selection
from nn.load import load_data

small = "CS170_Small_DataSet__13.txt"
large = "CS170_Large_DataSet__2.txt"

small_data = load_data(small)
large_data = load_data(large)

print('SMALL DATASET')
forward_selection(small_data)

print('\nLARGE DATASET')
forward_selection(large_data)
