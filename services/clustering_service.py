import numpy as np
from sklearn.cluster import KMeans
from services.preprocessing import preprocess_data
from utils.downloader import download_file

def run_clustering(file_url, n_clusters=3):
    try:
        file_path = download_file(file_url)

        df = preprocess_data(file_path)

        if df.empty:
            return {"error": "Empty dataset"}

        if df.shape[0] < n_clusters:
            return {"error": "Not enough samples for clustering"}

        df = df.loc[:, df.std() > 0]

        if df.shape[1] == 0:
            return {"error": "No valid features"}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df)

        clusters = np.nan_to_num(clusters)

        return {
            "clusters": clusters.tolist()
        }

    except Exception as e:
        return {"error": str(e)}