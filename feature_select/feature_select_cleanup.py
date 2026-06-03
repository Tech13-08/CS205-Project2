import os
import pandas as pd
import numpy as np

filename = "f1_strategy_dataset_v4.csv" 


df = pd.read_csv(filename)
df.columns = df.columns.str.strip()

features = [
    'TyreLife', 
    'LapNumber', 
    'Stint', 
    'LapTime (s)',
    'LapTime_Delta'
]
target = 'Position'

df_feature_select = df.dropna(subset=features + [target]).copy()

for col in features + [target]:
    df_feature_select[col] = pd.to_numeric(df_feature_select[col], errors='coerce')
df_feature_select = df_feature_select.dropna(subset=features + [target])


df_feature_select['class'] = np.where(df_feature_select[target] <= 3, 1.0, 2.0)

for col in features:
    df_feature_select[col] = (df_feature_select[col] - df_feature_select[col].mean()) / df_feature_select[col].std()

output = df_feature_select[['class'] + features].head(1500)

output_file = "feature_selection_data.txt"
output.to_csv(output_file, sep=' ', header=False, index=False)