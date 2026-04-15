from fastapi import APIRouter
from services.pca_service import run_pca
from services.volcano_service import run_volcano
from services.heatmap_service import run_heatmap

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