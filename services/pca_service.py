import numpy as np
from sklearn.decomposition import PCA
from services.preprocessing import preprocess_data
from utils.downloader import download_file

def run_pca(file_url):
    try:
        file_path = download_file(file_url)

        df = preprocess_data(file_path)

        if df.empty:
            return {"error": "Empty dataset"}

        if df.shape[1] < 2:
            return {"error": "Not enough features for PCA"}

        df = df.loc[:, df.std() > 0]

        if df.shape[1] < 2:
            return {"error": "Not enough variance for PCA"}

        pca = PCA(n_components=2)
        result = pca.fit_transform(df)

        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "pca": result.tolist()
        }

    except Exception as e:
        return {"error": str(e)}