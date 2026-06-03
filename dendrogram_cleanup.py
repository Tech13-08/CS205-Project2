import os
import pandas as pd
import numpy as np

filename = "f1_strategy_dataset_v4.csv" 


df = pd.read_csv(filename)
df.columns = df.columns.str.strip()

features = ['Driver', 'Compound', 'LapTime (s)', 'LapTime_Delta']
df = df.dropna(subset=features).copy()

df['LapTime (s)'] = pd.to_numeric(df['LapTime (s)'], errors='coerce')
df['LapTime_Delta'] = pd.to_numeric(df['LapTime_Delta'], errors='coerce')
df = df.dropna(subset=features)

sample_size = min(60, len(df))
df = df.sample(n=sample_size, random_state=42)

for col in ['LapTime (s)', 'LapTime_Delta']:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

output = df[['LapTime (s)', 'LapTime_Delta']].copy()

output.index = [
    f"{row['Driver']}_{row['Compound']}" for _, row in df.iterrows()
]

output_file = "dendrogram_data.csv"
output.to_csv(output_file)