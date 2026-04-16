import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from services.preprocessing import preprocess_data
from utils.downloader import download_file


def run_pca(file_url):
    try:
        file_path = download_file(file_url)
        df = preprocess_data(file_path)

        # ----------- VALIDATIONS -----------
        if df.empty:
            return {"error": "Empty dataset"}

        if df.shape[1] < 2:
            return {"error": "Not enough features for PCA"}

        # Remove zero variance columns
        df = df.loc[:, df.std() > 0]

        if df.shape[1] < 2:
            return {"error": "Not enough variance for PCA"}

        # ----------- SCALING -----------
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        # ----------- PCA -----------
        pca = PCA(n_components=2, random_state=42)
        result = pca.fit_transform(scaled_data)

        # ----------- CLUSTERING -----------
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(result)

        # ----------- CLEAN DATA -----------
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        # ----------- RESPONSE -----------
        return {
            "pca": result.tolist(),
            "variance": pca.explained_variance_ratio_.tolist(),
            "clusters": clusters.tolist(),
            "centroids": kmeans.cluster_centers_.tolist(),
            "n_samples": len(result)
        }

    except Exception as e:
        return {"error": str(e)}