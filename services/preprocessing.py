import pandas as pd
import numpy as np

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    df = df.fillna(df.mean(numeric_only=True))

    df = df.select_dtypes(include=[np.number])
    if df.empty:
        return df

    df = df.div(df.sum(axis=0), axis=1) * 1e6

    return df