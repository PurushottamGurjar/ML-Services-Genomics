from fastapi import APIRouter
from services.pca_service import run_pca
from services.volcano_service import run_volcano
from services.heatmap_service import run_heatmap
from services.clustering_service import run_clustering

router = APIRouter(prefix="/analyze")

@router.post("/pca")
def pca_api(data: dict):
    return run_pca(data["file_url"])

@router.post("/volcano")
def volcano_api(data: dict):
    return run_volcano(data["file_url"])

@router.post("/heatmap")
def heatmap_api(data: dict):
    return run_heatmap(data["file_url"])

@router.post("/clustering")
def clustering_api(data: dict):
    file_url = data["file_url"]
    n_clusters = data.get("n_clusters", 3)

    return run_clustering(file_url, n_clusters)