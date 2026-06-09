import pandas as pd
import numpy as np

def load_data(file):
    if file.endswith('.csv'):
        df = pd.read_csv(file, header=0, index_col=False)
        data = df.values
    else:
        data = np.loadtxt(file)
    return data