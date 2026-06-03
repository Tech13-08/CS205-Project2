import pandas as pd
import numpy as np

def load_data(file):
    if file.endswith('.csv'):
        df = pd.read_csv(file, header=None)
        data = df.select_dtypes(include=[np.number]).values
    else:
        data = np.loadtxt(file)
    return data