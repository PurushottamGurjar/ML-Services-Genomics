import numpy as np
from scipy import stats
from services.preprocessing import preprocess_data
from utils.downloader import download_file

def run_volcano(file_url):
    file_path = download_file(file_url)

    df = preprocess_data(file_path)

    if df.shape[1] < 2:
        return {"error": "Not enough columns for volcano"}

    mid = df.shape[1] // 2

    group1 = df.iloc[:, :mid]
    group2 = df.iloc[:, mid:]

    logFC = np.log2(group2.mean(axis=1) + 1e-9) - np.log2(group1.mean(axis=1) + 1e-9)

    pvals = stats.ttest_ind(group1.T, group2.T, nan_policy="omit").pvalue

    logFC = np.nan_to_num(logFC, nan=0.0, posinf=0.0, neginf=0.0)
    pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)

    return {
        "logFC": logFC.tolist(),
        "pvals": pvals.tolist()
    }