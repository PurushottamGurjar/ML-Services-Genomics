import numpy as np
from services.preprocessing import preprocess_data
from utils.downloader import download_file

def run_heatmap(file_url):
    try:
        file_path = download_file(file_url)

        df = preprocess_data(file_path)

        if df.empty:
            return {"error": "Empty dataset"}

        if df.shape[0] > 500:
            df = df.head(500)

        if df.shape[1] > 50:
            df = df.iloc[:, :50]

        matrix = np.nan_to_num(df.values, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "matrix": matrix.tolist(),
            "rows": df.index.astype(str).tolist(),
            "cols": df.columns.astype(str).tolist()
        }

    except Exception as e:
        return {"error": str(e)}