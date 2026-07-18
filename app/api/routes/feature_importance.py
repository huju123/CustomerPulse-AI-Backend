from fastapi import APIRouter
from app.models.schemas import FeatureImportanceResponse
from app.services import shap_service

router = APIRouter(
    prefix="/feature-importance",
    tags=["Explainability"]
)


@router.get("", response_model=FeatureImportanceResponse)
def get_feature_importance():
    """
    Returns global feature importance from the trained churn model (via SHAP).
    """
    features = shap_service.get_global_feature_importance()
    return FeatureImportanceResponse(
        model_name="XGBoost",
        features=features
    )